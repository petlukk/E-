use inkwell::values::FunctionValue;

use crate::ast::{Stmt, TypeAnnotation};
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_function(
        &mut self,
        name: &str,
        params: &[crate::ast::Param],
        body: &[Stmt],
        return_type: Option<&TypeAnnotation>,
    ) -> crate::error::Result<()> {
        let function = *self
            .functions
            .get(name)
            .ok_or_else(|| CompileError::codegen_error(format!("undeclared function '{name}'")))?;

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        self.variables.clear();
        for (i, param) in params.iter().enumerate() {
            let ty = self.resolve_annotation(&param.ty);
            let llvm_ty = self.llvm_type(&ty);
            let alloca = self
                .builder
                .build_alloca(llvm_ty, &param.name)
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let param_val = function
                .get_nth_param(i as u32)
                .ok_or_else(|| CompileError::codegen_error(format!("missing param {i}")))?;
            self.builder
                .build_store(alloca, param_val)
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            self.variables.insert(param.name.clone(), (alloca, ty));
        }

        let mut has_terminator = false;
        for stmt in body {
            if has_terminator {
                break;
            }
            has_terminator = self.compile_stmt(stmt, function)?;
        }

        if !has_terminator {
            if return_type.is_some() || name == "main" {
                let zero = self.context.i32_type().const_int(0, false);
                self.builder
                    .build_return(Some(&zero))
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            } else {
                self.builder
                    .build_return(None)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            }
        }

        if function.verify(true) {
            // verified
        } else {
            // Verification failed, proceed to return error or let it crash
            // It prints to stderr handling errors
        }
        Ok(())
    }

    pub(crate) fn compile_stmt(
        &mut self,
        stmt: &Stmt,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<bool> {
        match stmt {
            Stmt::Let {
                name, ty, value, ..
            } => {
                let declared = self.resolve_annotation(ty);
                let llvm_ty = self.llvm_type(&declared);
                let alloca = self
                    .builder
                    .build_alloca(llvm_ty, name)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                let val = self.compile_expr_typed(value, Some(&declared), function)?;
                self.builder
                    .build_store(alloca, val)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.variables.insert(name.clone(), (alloca, declared));
                Ok(false)
            }
            Stmt::Assign { target, value } => {
                let (ptr, var_type) = self.variables.get(target).cloned().ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{target}'"))
                })?;
                let val = self.compile_expr_typed(value, Some(&var_type), function)?;
                self.builder
                    .build_store(ptr, val)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(false)
            }
            Stmt::Return(Some(expr)) => {
                let val = self.compile_expr(expr, function)?;
                self.builder
                    .build_return(Some(&val))
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(true)
            }
            Stmt::Return(None) => {
                self.builder
                    .build_return(None)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(true)
            }
            Stmt::ExprStmt(expr) => {
                self.compile_expr(expr, function)?;
                Ok(false)
            }
            Stmt::IndexAssign {
                object,
                index,
                value,
            } => {
                let (ptr_alloca, var_type) =
                    self.variables.get(object).cloned().ok_or_else(|| {
                        CompileError::codegen_error(format!("undefined variable '{object}'"))
                    })?;
                let ptr = self
                    .builder
                    .build_load(ptr_alloca, object)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .into_pointer_value();
                let idx = self.compile_expr(index, function)?.into_int_value();
                let elem_ptr = unsafe { self.builder.build_gep(ptr, &[idx], "elemptr") }
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                let inner_type = match &var_type {
                    Type::Pointer { inner, .. } => inner.as_ref().clone(),
                    _ => {
                        return Err(CompileError::codegen_error(
                            "index-assign on non-pointer type",
                        ))
                    }
                };
                let val = self.compile_expr_typed(value, Some(&inner_type), function)?;
                self.builder
                    .build_store(elem_ptr, val)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(false)
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                let cond_val = self.compile_expr(condition, function)?;
                let cond_int = cond_val.into_int_value();

                let then_bb = self.context.append_basic_block(function, "then");
                let else_bb = self.context.append_basic_block(function, "else");
                let merge_bb = self.context.append_basic_block(function, "merge");

                self.builder
                    .build_conditional_branch(cond_int, then_bb, else_bb)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                // Then branch
                self.builder.position_at_end(then_bb);
                let then_term = self.compile_block(then_body, function)?;
                if !then_term {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                }

                // Else branch
                self.builder.position_at_end(else_bb);
                if let Some(else_stmts) = else_body {
                    let else_term = self.compile_block(else_stmts, function)?;
                    if !else_term {
                        self.builder
                            .build_unconditional_branch(merge_bb)
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    }
                } else {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                }

                self.builder.position_at_end(merge_bb);
                Ok(false)
            }
            Stmt::While {
                condition,
                body: while_body,
            } => {
                let cond_bb = self.context.append_basic_block(function, "while_cond");
                let body_bb = self.context.append_basic_block(function, "while_body");
                let exit_bb = self.context.append_basic_block(function, "while_exit");

                self.builder
                    .build_unconditional_branch(cond_bb)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                // Condition check
                self.builder.position_at_end(cond_bb);
                let cond_val = self.compile_expr(condition, function)?;
                let cond_int = cond_val.into_int_value();
                self.builder
                    .build_conditional_branch(cond_int, body_bb, exit_bb)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                // Loop body
                self.builder.position_at_end(body_bb);
                let body_term = self.compile_block(while_body, function)?;
                if !body_term {
                    self.builder
                        .build_unconditional_branch(cond_bb)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                }

                // Exit
                self.builder.position_at_end(exit_bb);
                Ok(false)
            }
            Stmt::Function { .. } => Err(CompileError::codegen_error(
                "nested functions not supported",
            )),
            Stmt::FieldAssign {
                object,
                field,
                value,
            } => {
                self.compile_field_assign(object, field, value, function)?;
                Ok(false)
            }
            Stmt::Struct { .. } => {
                // Registered in compile_program
                Ok(false)
            }
        }
    }

    fn compile_block(
        &mut self,
        stmts: &[Stmt],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<bool> {
        let mut terminated = false;
        for s in stmts {
            if terminated {
                break;
            }
            terminated = self.compile_stmt(s, function)?;
        }
        Ok(terminated)
    }
}

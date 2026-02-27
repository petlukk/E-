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
            let ty = Self::resolve_annotation(&param.ty);
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
            if name == "main" {
                let zero = self.context.i32_type().const_int(0, false);
                self.builder
                    .build_return(Some(&zero))
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            } else if let Some(ret_ann) = return_type {
                let ret_ty = Self::resolve_annotation(ret_ann);
                let llvm_ty = self.llvm_type(&ret_ty);
                let zero_val = llvm_ty.const_zero();
                self.builder
                    .build_return(Some(&zero_val))
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            } else {
                self.builder
                    .build_return(None)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            }
        }

        if !function.verify(true) {
            return Err(CompileError::codegen_error(format!(
                "LLVM verification failed for function '{name}'"
            )));
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
                // span ignored via ..
                let declared = Self::resolve_annotation(ty);
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
            Stmt::Assign { target, value, .. } => {
                let (ptr, var_type) = self.variables.get(target).cloned().ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{target}'"))
                })?;
                let val = self.compile_expr_typed(value, Some(&var_type), function)?;
                self.builder
                    .build_store(ptr, val)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(false)
            }
            Stmt::Return(Some(expr), _) => {
                let ret_hint = function
                    .get_name()
                    .to_str()
                    .ok()
                    .and_then(|n| self.func_signatures.get(n))
                    .and_then(|(_, ret)| ret.clone());
                let val = self.compile_expr_typed(expr, ret_hint.as_ref(), function)?;
                self.builder
                    .build_return(Some(&val))
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(true)
            }
            Stmt::Return(None, _) => {
                self.builder
                    .build_return(None)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(true)
            }
            Stmt::ExprStmt(expr, _) => {
                self.compile_expr(expr, function)?;
                Ok(false)
            }
            Stmt::IndexAssign {
                object,
                index,
                value,
                ..
            } => {
                let (ptr_alloca, var_type) =
                    self.variables.get(object).cloned().ok_or_else(|| {
                        CompileError::codegen_error(format!("undefined variable '{object}'"))
                    })?;
                let ptr_ty = self.llvm_type(&var_type);
                let ptr = self
                    .builder
                    .build_load(ptr_ty, ptr_alloca, object)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .into_pointer_value();
                let idx = self.compile_expr(index, function)?.into_int_value();
                let inner_type = match &var_type {
                    Type::Pointer { inner, .. } => inner.as_ref().clone(),
                    _ => {
                        return Err(CompileError::codegen_error(
                            "index-assign on non-pointer type",
                        ))
                    }
                };
                let inner_llvm_ty = self.llvm_type(&inner_type);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(inner_llvm_ty, ptr, &[idx], "elemptr")
                }
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
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
                ..
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
                ..
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
            Stmt::ForEach {
                var,
                start,
                end,
                body: foreach_body,
                ..
            } => {
                let cond_bb = self.context.append_basic_block(function, "foreach_cond");
                let body_bb = self.context.append_basic_block(function, "foreach_body");
                let exit_bb = self.context.append_basic_block(function, "foreach_exit");

                let start_val = self.compile_expr(start, function)?.into_int_value();
                let end_val = self.compile_expr(end, function)?.into_int_value();

                let entry_bb = self.builder.get_insert_block().unwrap();
                self.builder
                    .build_unconditional_branch(cond_bb)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                // Alloca in function entry block (not loop block) to avoid
                // stack growth at O0 where mem2reg does not run.
                let i32_type = self.context.i32_type();
                let fn_entry = function.get_first_basic_block().unwrap();
                let alloca = if let Some(first_instr) = fn_entry.get_first_instruction() {
                    self.builder.position_before(&first_instr);
                    let a = self
                        .builder
                        .build_alloca(i32_type, var)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    self.builder.position_at_end(cond_bb);
                    a
                } else {
                    self.builder.position_at_end(cond_bb);
                    self.builder
                        .build_alloca(i32_type, var)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?
                };

                // Condition block with phi node
                let phi = self
                    .builder
                    .build_phi(i32_type, var)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                phi.add_incoming(&[(&start_val, entry_bb)]);

                let i_val = phi.as_basic_value().into_int_value();

                self.builder
                    .build_store(alloca, i_val)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.variables
                    .insert(var.clone(), (alloca, crate::typeck::Type::I32));

                let cond = self
                    .builder
                    .build_int_compare(inkwell::IntPredicate::SLT, i_val, end_val, "foreach_cmp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.builder
                    .build_conditional_branch(cond, body_bb, exit_bb)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                // Body
                self.builder.position_at_end(body_bb);
                let body_term = self.compile_block(foreach_body, function)?;
                if !body_term {
                    let i_current = self
                        .builder
                        .build_load(i32_type, alloca, "i_cur")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?
                        .into_int_value();
                    let one = i32_type.const_int(1, false);
                    let i_next = self
                        .builder
                        .build_int_add(i_current, one, "i_next")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                    self.builder
                        .build_store(alloca, i_next)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

                    let body_end_bb = self.builder.get_insert_block().unwrap();
                    phi.add_incoming(&[(&i_next, body_end_bb)]);

                    self.builder
                        .build_unconditional_branch(cond_bb)
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                }

                self.builder.position_at_end(exit_bb);
                Ok(false)
            }
            Stmt::Unroll { body, .. } => {
                // span ignored via ..
                // Compile the inner loop normally — LLVM at O2/O3 handles unrolling.
                // The unroll(N) annotation is a semantic hint; metadata attachment
                // requires inkwell API support for instruction-level metadata which
                // is not yet available in inkwell 0.5. For now, the loop compiles
                // correctly and LLVM's heuristics apply.
                self.compile_stmt(body, function)
            }
            Stmt::Function { .. } => Err(CompileError::codegen_error(
                "nested functions not supported",
            )),
            Stmt::FieldAssign {
                object,
                field,
                value,
                ..
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

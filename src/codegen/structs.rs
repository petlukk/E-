use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_struct_literal(
        &mut self,
        name: &str,
        fields: &[(String, Expr)],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let struct_type = *self
            .struct_types
            .get(name)
            .ok_or_else(|| CompileError::codegen_error(format!("unknown struct type '{name}'")))?;
        let field_map = self.struct_fields.get(name).cloned().ok_or_else(|| {
            CompileError::codegen_error(format!("unknown struct fields for '{name}'"))
        })?;
        let alloca = self
            .builder
            .build_alloca(struct_type, "struct_tmp")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        for (field_name, field_expr) in fields {
            let (idx, field_type) = field_map
                .iter()
                .find(|(n, _, _)| n == field_name)
                .map(|(_, i, t)| (*i, t.clone()))
                .ok_or_else(|| {
                    CompileError::codegen_error(format!(
                        "struct '{name}' has no field '{field_name}'"
                    ))
                })?;
            let val = self.compile_expr_typed(field_expr, Some(&field_type), function)?;
            let field_ptr = self
                .builder
                .build_struct_gep(struct_type, alloca, idx, &format!("{name}.{field_name}"))
                .map_err(|e| CompileError::codegen_error(format!("struct gep error: {e}")))?;
            self.builder
                .build_store(field_ptr, val)
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }

        let loaded = self
            .builder
            .build_load(struct_type, alloca, name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(loaded)
    }

    pub(crate) fn compile_field_access(
        &mut self,
        object: &Expr,
        field: &str,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let (struct_ptr, struct_name) = self.resolve_struct_ptr(object, function)?;
        let struct_type = *self.struct_types.get(&struct_name).ok_or_else(|| {
            CompileError::codegen_error(format!("unknown struct type '{struct_name}'"))
        })?;
        let field_map = self.struct_fields.get(&struct_name).ok_or_else(|| {
            CompileError::codegen_error(format!("unknown struct '{struct_name}'"))
        })?;
        let (idx, field_type) = field_map
            .iter()
            .find(|(n, _, _)| n == field)
            .map(|(_, i, t)| (*i, t.clone()))
            .ok_or_else(|| {
                CompileError::codegen_error(format!(
                    "struct '{struct_name}' has no field '{field}'"
                ))
            })?;
        let field_ptr = self
            .builder
            .build_struct_gep(
                struct_type,
                struct_ptr,
                idx,
                &format!("{struct_name}.{field}"),
            )
            .map_err(|e| CompileError::codegen_error(format!("struct gep error: {e}")))?;
        let field_llvm_ty = self.llvm_type(&field_type);
        let val = self
            .builder
            .build_load(field_llvm_ty, field_ptr, field)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(val)
    }

    pub(crate) fn compile_field_assign(
        &mut self,
        object: &Expr,
        field: &str,
        value: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<()> {
        let (struct_ptr, struct_name) = self.resolve_struct_ptr(object, function)?;
        let struct_type = *self.struct_types.get(&struct_name).ok_or_else(|| {
            CompileError::codegen_error(format!("unknown struct type '{struct_name}'"))
        })?;
        let field_map = self.struct_fields.get(&struct_name).ok_or_else(|| {
            CompileError::codegen_error(format!("unknown struct '{struct_name}'"))
        })?;
        let (idx, field_type) = field_map
            .iter()
            .find(|(n, _, _)| n == field)
            .map(|(_, i, t)| (*i, t.clone()))
            .ok_or_else(|| {
                CompileError::codegen_error(format!(
                    "struct '{struct_name}' has no field '{field}'"
                ))
            })?;
        let field_ptr = self
            .builder
            .build_struct_gep(
                struct_type,
                struct_ptr,
                idx,
                &format!("{struct_name}.{field}"),
            )
            .map_err(|e| CompileError::codegen_error(format!("struct gep error: {e}")))?;
        let val = self.compile_expr_typed(value, Some(&field_type), function)?;
        self.builder
            .build_store(field_ptr, val)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(())
    }

    fn resolve_struct_ptr(
        &mut self,
        object: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<(PointerValue<'ctx>, String)> {
        match object {
            Expr::Variable(name, _) => {
                let (alloca, ty) = self.variables.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{name}'"))
                })?;
                match ty {
                    Type::Struct(sn) => Ok((*alloca, sn.clone())),
                    Type::Pointer { inner, .. } => match inner.as_ref() {
                        Type::Struct(sn) => {
                            let ptr_ty = self.llvm_type(ty);
                            let ptr = self
                                .builder
                                .build_load(ptr_ty, *alloca, name)
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                                .into_pointer_value();
                            Ok((ptr, sn.clone()))
                        }
                        _ => Err(CompileError::codegen_error(format!(
                            "pointer does not point to struct: {ty:?}"
                        ))),
                    },
                    _ => Err(CompileError::codegen_error(format!(
                        "field access on non-struct type: {ty:?}"
                    ))),
                }
            }
            Expr::Index {
                object: arr, index, ..
            } => {
                let arr_val = self.compile_expr(arr, function)?;
                let idx = self.compile_expr(index, function)?.into_int_value();
                let ptr = arr_val.into_pointer_value();
                let struct_name = self.resolve_struct_name_from_expr(arr)?;
                let struct_type = *self.struct_types.get(&struct_name).ok_or_else(|| {
                    CompileError::codegen_error(format!("unknown struct type '{struct_name}'"))
                })?;
                let elem_ptr =
                    unsafe { self.builder.build_gep(struct_type, ptr, &[idx], "elemptr") }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok((elem_ptr, struct_name))
            }
            Expr::FieldAccess { .. } => Err(CompileError::codegen_error(
                // span absorbed by ..
                "nested struct field access not supported",
            )),
            _ => Err(CompileError::codegen_error(
                "unsupported object for field access",
            )),
        }
    }

    fn resolve_struct_name_from_expr(&self, expr: &Expr) -> crate::error::Result<String> {
        match expr {
            Expr::Variable(name, _) => {
                let (_, ty) = self.variables.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{name}'"))
                })?;
                match ty {
                    Type::Pointer { inner, .. } => match inner.as_ref() {
                        Type::Struct(sn) => Ok(sn.clone()),
                        _ => Err(CompileError::codegen_error(
                            "pointer does not point to struct",
                        )),
                    },
                    _ => Err(CompileError::codegen_error("expected pointer to struct")),
                }
            }
            _ => Err(CompileError::codegen_error(
                "cannot resolve struct name from expression",
            )),
        }
    }
}

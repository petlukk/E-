#[cfg(feature = "llvm")]
mod expressions;
#[cfg(feature = "llvm")]
mod statements;

#[cfg(feature = "llvm")]
use std::collections::HashMap;

#[cfg(feature = "llvm")]
use inkwell::builder::Builder;
#[cfg(feature = "llvm")]
use inkwell::context::Context;
#[cfg(feature = "llvm")]
use inkwell::module::{Linkage, Module};
#[cfg(feature = "llvm")]
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
#[cfg(feature = "llvm")]
use inkwell::values::{FunctionValue, PointerValue};
#[cfg(feature = "llvm")]
use inkwell::AddressSpace;

#[cfg(feature = "llvm")]
use crate::ast::{Stmt, TypeAnnotation};
#[cfg(feature = "llvm")]
use crate::typeck::Type;

#[cfg(feature = "llvm")]
pub struct CodeGenerator<'ctx> {
    pub(crate) context: &'ctx Context,
    pub(crate) module: Module<'ctx>,
    pub(crate) builder: Builder<'ctx>,
    pub(crate) variables: HashMap<String, (PointerValue<'ctx>, Type)>,
    pub(crate) functions: HashMap<String, FunctionValue<'ctx>>,
}

#[cfg(feature = "llvm")]
impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        Self {
            context,
            module,
            builder,
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn compile_program(&mut self, stmts: &[Stmt]) -> crate::error::Result<()> {
        self.declare_printf();

        for stmt in stmts {
            if let Stmt::Function {
                name,
                params,
                return_type,
                export,
                ..
            } = stmt
            {
                self.declare_function(name, params, return_type.as_ref(), *export)?;
            }
        }

        for stmt in stmts {
            if let Stmt::Function {
                name,
                params,
                body,
                return_type,
                ..
            } = stmt
            {
                self.compile_function(name, params, body, return_type.as_ref())?;
            }
        }

        Ok(())
    }

    fn declare_printf(&mut self) {
        let i32_type = self.context.i32_type();
        let i8_ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
        let printf_type = i32_type.fn_type(&[BasicMetadataTypeEnum::from(i8_ptr_type)], true);
        let printf = self
            .module
            .add_function("printf", printf_type, Some(Linkage::External));
        self.functions.insert("printf".to_string(), printf);
    }

    pub(crate) fn llvm_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::I32 | Type::IntLiteral => BasicTypeEnum::IntType(self.context.i32_type()),
            Type::I64 => BasicTypeEnum::IntType(self.context.i64_type()),
            Type::F32 => BasicTypeEnum::FloatType(self.context.f32_type()),
            Type::F64 | Type::FloatLiteral => BasicTypeEnum::FloatType(self.context.f64_type()),
            Type::Bool => BasicTypeEnum::IntType(self.context.bool_type()),
            Type::Pointer { inner, .. } => {
                let inner_ty = self.llvm_type(inner);
                BasicTypeEnum::PointerType(inner_ty.ptr_type(AddressSpace::default()))
            }
            _ => BasicTypeEnum::IntType(self.context.i32_type()),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    pub(crate) fn resolve_annotation(&self, ann: &TypeAnnotation) -> Type {
        match ann {
            TypeAnnotation::Named(name) => match name.as_str() {
                "i32" => Type::I32,
                "i64" => Type::I64,
                "f32" => Type::F32,
                "f64" => Type::F64,
                "bool" => Type::Bool,
                _ => Type::I32,
            },
            TypeAnnotation::Pointer { mutable, inner } => {
                let inner_type = self.resolve_annotation(inner);
                Type::Pointer {
                    mutable: *mutable,
                    inner: Box::new(inner_type),
                }
            }
        }
    }

    fn declare_function(
        &mut self,
        name: &str,
        params: &[crate::ast::Param],
        return_type: Option<&TypeAnnotation>,
        export: bool,
    ) -> crate::error::Result<()> {
        let param_types: Vec<BasicMetadataTypeEnum> = params
            .iter()
            .map(|p| {
                let ty = self.resolve_annotation(&p.ty);
                self.llvm_type(&ty).into()
            })
            .collect();

        let fn_type = match return_type {
            Some(ann) => {
                let ret_ty = self.resolve_annotation(ann);
                match self.llvm_type(&ret_ty) {
                    BasicTypeEnum::IntType(t) => t.fn_type(&param_types, false),
                    BasicTypeEnum::FloatType(t) => t.fn_type(&param_types, false),
                    _ => self.context.i32_type().fn_type(&param_types, false),
                }
            }
            None => {
                if name == "main" {
                    self.context.i32_type().fn_type(&param_types, false)
                } else {
                    self.context.void_type().fn_type(&param_types, false)
                }
            }
        };

        let linkage = if export || name == "main" {
            Some(Linkage::External)
        } else {
            Some(Linkage::Private)
        };

        let function = self.module.add_function(name, fn_type, linkage);
        self.functions.insert(name.to_string(), function);
        Ok(())
    }

    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    pub fn print_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }
}

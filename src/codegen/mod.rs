#[cfg(feature = "llvm")]
mod expressions;
#[cfg(feature = "llvm")]
mod simd;
#[cfg(feature = "llvm")]
mod simd_arithmetic;
#[cfg(feature = "llvm")]
mod simd_math;
#[cfg(feature = "llvm")]
mod simd_memory;
#[cfg(feature = "llvm")]
mod statements;
#[cfg(feature = "llvm")]
mod structs;

#[cfg(feature = "llvm")]
use std::collections::HashMap;

#[cfg(feature = "llvm")]
use inkwell::builder::Builder;
#[cfg(feature = "llvm")]
use inkwell::context::Context;
#[cfg(feature = "llvm")]
use inkwell::module::{Linkage, Module};
#[cfg(feature = "llvm")]
use inkwell::DLLStorageClass;
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
    pub(crate) struct_types: HashMap<String, inkwell::types::StructType<'ctx>>,
    pub(crate) struct_fields: HashMap<String, Vec<(String, u32, Type)>>,
}

#[cfg(feature = "llvm")]
impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        // On Windows, LLVM emits a reference to _fltused in every object that
        // uses floating-point.  It is an MSVC CRT sentinel (int = 1) that is
        // never called at runtime.  Defining it here satisfies lld-link without
        // /FORCE:UNRESOLVED and without linking against the CRT.
        #[cfg(target_os = "windows")]
        {
            let i32_ty = context.i32_type();
            let fltused = module.add_global(i32_ty, Some(AddressSpace::default()), "_fltused");
            fltused.set_initializer(&i32_ty.const_int(1, false));
            fltused.set_linkage(Linkage::External);
        }

        Self {
            context,
            module,
            builder,
            variables: HashMap::new(),
            functions: HashMap::new(),
            struct_types: HashMap::new(),
            struct_fields: HashMap::new(),
        }
    }

    pub fn compile_program(&mut self, stmts: &[Stmt]) -> crate::error::Result<()> {
        self.declare_printf();

        for stmt in stmts {
            if let Stmt::Struct { name, fields } = stmt {
                self.register_struct(name, fields);
            }
        }

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
        let i8_ptr_type = self.context.ptr_type(AddressSpace::default());
        let printf_type = i32_type.fn_type(&[BasicMetadataTypeEnum::from(i8_ptr_type)], true);
        let printf = self
            .module
            .add_function("printf", printf_type, Some(Linkage::External));
        self.functions.insert("printf".to_string(), printf);
    }

    fn register_struct(&mut self, name: &str, fields: &[crate::ast::StructField]) {
        let field_types: Vec<BasicTypeEnum> = fields
            .iter()
            .map(|f| {
                let ty = self.resolve_annotation(&f.ty);
                self.llvm_type(&ty)
            })
            .collect();
        let struct_type = self.context.struct_type(&field_types, false);
        self.struct_types.insert(name.to_string(), struct_type);
        let field_map: Vec<(String, u32, Type)> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let ty = self.resolve_annotation(&f.ty);
                (f.name.clone(), i as u32, ty)
            })
            .collect();
        self.struct_fields.insert(name.to_string(), field_map);
    }

    pub(crate) fn llvm_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::I8 | Type::U8 => BasicTypeEnum::IntType(self.context.i8_type()),
            Type::I16 | Type::U16 => BasicTypeEnum::IntType(self.context.i16_type()),
            Type::I32 | Type::IntLiteral => BasicTypeEnum::IntType(self.context.i32_type()),
            Type::I64 => BasicTypeEnum::IntType(self.context.i64_type()),
            Type::F32 => BasicTypeEnum::FloatType(self.context.f32_type()),
            Type::F64 | Type::FloatLiteral => BasicTypeEnum::FloatType(self.context.f64_type()),
            Type::Bool => BasicTypeEnum::IntType(self.context.bool_type()),
            Type::Pointer { .. } => {
                BasicTypeEnum::PointerType(self.context.ptr_type(AddressSpace::default()))
            }
            Type::Vector { elem, width } => {
                let elem_ty = self.llvm_type(elem);
                match elem_ty {
                    BasicTypeEnum::IntType(t) => t.vec_type(*width as u32).into(),
                    BasicTypeEnum::FloatType(t) => t.vec_type(*width as u32).into(),
                    BasicTypeEnum::PointerType(t) => t.vec_type(*width as u32).into(),
                    _ => self.context.i32_type().vec_type(*width as u32).into(),
                }
            }
            Type::Struct(name) => {
                if let Some(st) = self.struct_types.get(name) {
                    BasicTypeEnum::StructType(*st)
                } else {
                    BasicTypeEnum::IntType(self.context.i32_type())
                }
            }
            _ => BasicTypeEnum::IntType(self.context.i32_type()),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    pub(crate) fn resolve_annotation(&self, ann: &TypeAnnotation) -> Type {
        match ann {
            TypeAnnotation::Named(name) => match name.as_str() {
                "i8" => Type::I8,
                "u8" => Type::U8,
                "i16" => Type::I16,
                "u16" => Type::U16,
                "i32" => Type::I32,
                "i64" => Type::I64,
                "f32" => Type::F32,
                "f64" => Type::F64,
                "bool" => Type::Bool,
                other => Type::Struct(other.to_string()),
            },
            TypeAnnotation::Pointer {
                mutable,
                restrict,
                inner,
            } => {
                let inner_type = self.resolve_annotation(inner);
                Type::Pointer {
                    mutable: *mutable,
                    restrict: *restrict,
                    inner: Box::new(inner_type),
                }
            }
            TypeAnnotation::Vector { elem, width } => {
                let elem_type = self.resolve_annotation(elem);
                Type::Vector {
                    elem: Box::new(elem_type),
                    width: *width,
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
                let llvm_ret = self.llvm_type(&ret_ty);
                llvm_ret.fn_type(&param_types, false)
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

        // On Windows PE/COFF, ExternalLinkage alone does not place a symbol in
        // the DLL export table — dllexport is required for that.  On Linux/ELF
        // ExternalLinkage is sufficient; this attribute is a harmless no-op there.
        if export {
            function.as_global_value().set_dll_storage_class(DLLStorageClass::Export);
        }

        for (i, param) in params.iter().enumerate() {
            if let TypeAnnotation::Pointer { restrict: true, .. } = &param.ty {
                let kind_id = inkwell::attributes::Attribute::get_named_enum_kind_id("noalias");
                let attr = self.context.create_enum_attribute(kind_id, 0);
                function.add_attribute(inkwell::attributes::AttributeLoc::Param(i as u32), attr);
            }
        }

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

use crate::ast::*;
use crate::error::CompileError;
use crate::lexer::Span;

/// Desugar `Stmt::Kernel` into `Stmt::Function` with a generated loop.
///
/// After this pass, the AST contains only `Stmt::Function` at the top level
/// (plus `Stmt::Struct` and `Stmt::Const`). All downstream phases — type
/// checking, codegen, metadata, header, bindings — handle functions only.
pub fn desugar_kernels(stmts: Vec<Stmt>) -> crate::error::Result<Vec<Stmt>> {
    stmts
        .into_iter()
        .map(|stmt| {
            if matches!(stmt, Stmt::Kernel { .. }) {
                desugar_kernel(stmt)
            } else {
                Ok(stmt)
            }
        })
        .collect()
}

fn desugar_kernel(kernel: Stmt) -> crate::error::Result<Stmt> {
    let Stmt::Kernel {
        name,
        mut params,
        range_var,
        range_bound,
        step,
        tail,
        tail_body,
        body,
        export,
        span,
    } = kernel
    else {
        unreachable!()
    };

    // Validate: range_bound must not collide with any declared param
    for p in &params {
        if p.name == range_bound {
            return Err(CompileError::parse_error(
                format!("range bound '{range_bound}' collides with parameter name"),
                span.start.clone(),
            ));
        }
    }

    // Validate: no assignment to the iteration variable in user body
    check_no_assign_to_var(&body, &range_var)?;
    if let Some(ref tb) = tail_body {
        check_no_assign_to_var(tb, &range_var)?;
    }

    let s = span.clone();

    // Append the range bound as an i32 parameter
    params.push(Param {
        name: range_bound.clone(),
        ty: TypeAnnotation::Named("i32".to_string(), s.clone()),
        output: false,
        cap: None,
        count: None,
        span: s.clone(),
    });

    // let mut VAR: i32 = 0
    let let_var = Stmt::Let {
        name: range_var.clone(),
        ty: TypeAnnotation::Named("i32".to_string(), s.clone()),
        value: Expr::Literal(Literal::Integer(0), s.clone()),
        mutable: true,
        span: s.clone(),
    };

    let main_loop = build_main_loop(&range_var, &range_bound, step, body, &s);
    let mut func_body = vec![let_var, main_loop];

    // Append tail handling
    append_tail(
        &mut func_body,
        tail,
        tail_body,
        &range_var,
        &range_bound,
        &s,
    );

    Ok(Stmt::Function {
        name,
        params,
        return_type: None,
        body: func_body,
        export,
        span,
    })
}

/// Build: `while VAR + STEP <= RANGE { BODY; VAR = VAR + STEP }`
fn build_main_loop(
    range_var: &str,
    range_bound: &str,
    step: u32,
    mut body: Vec<Stmt>,
    s: &Span,
) -> Stmt {
    let cond = Expr::Binary(
        Box::new(Expr::Binary(
            Box::new(Expr::Variable(range_var.to_string(), s.clone())),
            BinaryOp::Add,
            Box::new(Expr::Literal(Literal::Integer(step as i64), s.clone())),
            s.clone(),
        )),
        BinaryOp::LessEqual,
        Box::new(Expr::Variable(range_bound.to_string(), s.clone())),
        s.clone(),
    );

    body.push(make_increment(range_var, step as i64, s));

    Stmt::While {
        condition: cond,
        body,
        span: s.clone(),
    }
}

/// Append tail loop/guard based on strategy.
fn append_tail(
    func_body: &mut Vec<Stmt>,
    tail: Option<TailStrategy>,
    tail_body: Option<Vec<Stmt>>,
    range_var: &str,
    range_bound: &str,
    s: &Span,
) {
    match tail {
        Some(TailStrategy::Scalar) => {
            if let Some(mut tb) = tail_body {
                // while VAR < RANGE { TAIL_BODY; VAR = VAR + 1 }
                let cond = Expr::Binary(
                    Box::new(Expr::Variable(range_var.to_string(), s.clone())),
                    BinaryOp::Less,
                    Box::new(Expr::Variable(range_bound.to_string(), s.clone())),
                    s.clone(),
                );
                tb.push(make_increment(range_var, 1, s));
                func_body.push(Stmt::While {
                    condition: cond,
                    body: tb,
                    span: s.clone(),
                });
            }
        }
        Some(TailStrategy::Mask) => {
            if let Some(tb) = tail_body {
                // if VAR < RANGE { TAIL_BODY }
                let cond = Expr::Binary(
                    Box::new(Expr::Variable(range_var.to_string(), s.clone())),
                    BinaryOp::Less,
                    Box::new(Expr::Variable(range_bound.to_string(), s.clone())),
                    s.clone(),
                );
                func_body.push(Stmt::If {
                    condition: cond,
                    then_body: tb,
                    else_body: None,
                    span: s.clone(),
                });
            }
        }
        Some(TailStrategy::Pad) | None => {}
    }
}

/// Build: `VAR = VAR + amount`
fn make_increment(var: &str, amount: i64, s: &Span) -> Stmt {
    Stmt::Assign {
        target: var.to_string(),
        value: Expr::Binary(
            Box::new(Expr::Variable(var.to_string(), s.clone())),
            BinaryOp::Add,
            Box::new(Expr::Literal(Literal::Integer(amount), s.clone())),
            s.clone(),
        ),
        span: s.clone(),
    }
}

/// Recursively checks that no statement assigns to the given variable.
fn check_no_assign_to_var(stmts: &[Stmt], var: &str) -> crate::error::Result<()> {
    for stmt in stmts {
        match stmt {
            Stmt::Assign { target, span, .. } if target == var => {
                return Err(CompileError::type_error(
                    format!(
                        "cannot assign to loop variable '{var}' \
                         — it is advanced by the kernel's step"
                    ),
                    span.clone(),
                ));
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                check_no_assign_to_var(then_body, var)?;
                if let Some(eb) = else_body {
                    check_no_assign_to_var(eb, var)?;
                }
            }
            Stmt::While { body, .. } => check_no_assign_to_var(body, var)?,
            Stmt::ForEach { body, .. } => check_no_assign_to_var(body, var)?,
            Stmt::Unroll { body, .. } => check_no_assign_to_var(&[*body.clone()], var)?,
            _ => {}
        }
    }
    Ok(())
}

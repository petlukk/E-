use crate::ast::Stmt;

pub fn generate_json(stmts: &[Stmt], lib_name: &str) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"library\": \"{lib_name}\",\n"));

    // Collect exported functions
    out.push_str("  \"exports\": [");
    let mut first_export = true;
    for stmt in stmts {
        if let Stmt::Function {
            name,
            params,
            return_type,
            export: true,
            ..
        } = stmt
        {
            if !first_export {
                out.push(',');
            }
            first_export = false;
            out.push_str("\n    {\n");
            out.push_str(&format!("      \"name\": \"{name}\",\n"));
            out.push_str("      \"args\": [");
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let ty_str = format!("{}", p.ty);
                let direction = if p.output { "out" } else { "in" };
                let cap_str = match &p.cap {
                    Some(c) => format!("\"{}\"", json_escape(c)),
                    None => "null".to_string(),
                };
                let count_str = match &p.count {
                    Some(c) => format!("\"{}\"", json_escape(c)),
                    None => "null".to_string(),
                };
                out.push_str(&format!(
                    "{{\"name\": \"{}\", \"type\": \"{}\", \"direction\": \"{}\", \"cap\": {}, \"count\": {}}}",
                    p.name,
                    json_escape(&ty_str),
                    direction,
                    cap_str,
                    count_str
                ));
            }
            out.push_str("],\n");
            match return_type {
                Some(ty) => {
                    let ty_str = format!("{ty}");
                    out.push_str(&format!(
                        "      \"return_type\": \"{}\"\n",
                        json_escape(&ty_str)
                    ));
                }
                None => out.push_str("      \"return_type\": null\n"),
            }
            out.push_str("    }");
        }
    }
    out.push_str("\n  ],\n");

    // Collect structs
    out.push_str("  \"structs\": [");
    let mut first_struct = true;
    for stmt in stmts {
        if let Stmt::Struct { name, fields, .. } = stmt {
            if !first_struct {
                out.push(',');
            }
            first_struct = false;
            out.push_str("\n    {\n");
            out.push_str(&format!("      \"name\": \"{name}\",\n"));
            out.push_str("      \"fields\": [");
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let ty_str = format!("{}", f.ty);
                out.push_str(&format!(
                    "{{\"name\": \"{}\", \"type\": \"{}\"}}",
                    f.name,
                    json_escape(&ty_str)
                ));
            }
            out.push_str("]\n");
            out.push_str("    }");
        }
    }
    out.push_str("\n  ]\n");

    out.push_str("}\n");
    out
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

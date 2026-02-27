/// Shared types and JSON parsing for all `ea bind` generators.

pub struct ExportFunc {
    pub name: String,
    pub args: Vec<Arg>,
    pub return_type: Option<String>,
}

pub struct Arg {
    pub name: String,
    pub ty: String,
}

/// Returns true if `ty` is a pointer type (`*T`, `*mut T`, `*restrict T`, etc.).
pub fn is_pointer(ty: &str) -> bool {
    pointer_inner(ty).is_some()
}

/// Returns true if `ty` is a mutable pointer (`*mut T` or `*restrict mut T`).
pub fn is_mut_pointer(ty: &str) -> bool {
    let ty = ty.trim();
    ty.starts_with("*mut ") || ty.starts_with("*restrict mut ")
}

/// Strips the pointer prefix and returns the inner type, e.g. `"*mut f32"` → `"f32"`.
pub fn pointer_inner(ty: &str) -> Option<&str> {
    let ty = ty.trim();
    if let Some(rest) = ty.strip_prefix("*mut ") {
        Some(rest.trim())
    } else if let Some(rest) = ty.strip_prefix("*restrict mut ") {
        Some(rest.trim())
    } else if let Some(rest) = ty.strip_prefix("*restrict ") {
        Some(rest.trim())
    } else if let Some(rest) = ty.strip_prefix('*') {
        Some(rest.trim())
    } else {
        None
    }
}

pub fn is_integer_type(ty: &str) -> bool {
    matches!(ty, "i32" | "i64" | "u32" | "u64")
}

/// Identify length parameters that should be auto-filled from a preceding pointer/slice.
pub fn find_collapsed_args(args: &[Arg]) -> Vec<bool> {
    let length_names = ["n", "len", "length", "count", "size", "num"];
    let mut collapsed = vec![false; args.len()];
    let mut has_preceding_pointer = false;

    for (i, arg) in args.iter().enumerate() {
        if pointer_inner(&arg.ty).is_some() {
            has_preceding_pointer = true;
        } else if has_preceding_pointer
            && is_integer_type(&arg.ty)
            && length_names.contains(&arg.name.as_str())
        {
            collapsed[i] = true;
        }
    }
    collapsed
}

// --- Minimal JSON parsing (no serde) ---

pub fn parse_exports(json: &str) -> Result<Vec<ExportFunc>, String> {
    let exports_start = json
        .find("\"exports\"")
        .ok_or("missing \"exports\" in JSON")?;
    let after_key = &json[exports_start + "\"exports\"".len()..];
    let arr_start = after_key.find('[').ok_or("missing '[' after \"exports\"")?;
    let arr_content = &after_key[arr_start..];
    let arr_end = find_matching_bracket(arr_content)?;
    let arr_str = &arr_content[1..arr_end];

    let mut funcs = Vec::new();
    let mut pos = 0;
    while pos < arr_str.len() {
        if let Some(obj_start) = arr_str[pos..].find('{') {
            let obj_slice = &arr_str[pos + obj_start..];
            let obj_end = find_matching_brace(obj_slice)?;
            let obj_str = &obj_slice[..=obj_end];
            funcs.push(parse_export_obj(obj_str)?);
            pos = pos + obj_start + obj_end + 1;
        } else {
            break;
        }
    }
    Ok(funcs)
}

fn parse_export_obj(obj: &str) -> Result<ExportFunc, String> {
    let name = parse_string_field(obj, "name").ok_or("missing \"name\" in export")?;
    let return_type = parse_nullable_string_field(obj, "return_type");

    let args_start = obj.find("\"args\"").ok_or("missing \"args\" in export")?;
    let after_key = &obj[args_start + "\"args\"".len()..];
    let arr_start = after_key.find('[').ok_or("missing '[' after \"args\"")?;
    let arr_content = &after_key[arr_start..];
    let arr_end = find_matching_bracket(arr_content)?;
    let arr_str = &arr_content[1..arr_end];

    let mut args = Vec::new();
    let mut pos = 0;
    while pos < arr_str.len() {
        if let Some(obj_start) = arr_str[pos..].find('{') {
            let obj_slice = &arr_str[pos + obj_start..];
            let obj_end = find_matching_brace(obj_slice)?;
            let arg_obj = &obj_slice[..=obj_end];
            let arg_name = parse_string_field(arg_obj, "name").ok_or("missing \"name\" in arg")?;
            let arg_type = parse_string_field(arg_obj, "type").ok_or("missing \"type\" in arg")?;
            args.push(Arg {
                name: arg_name,
                ty: arg_type,
            });
            pos = pos + obj_start + obj_end + 1;
        } else {
            break;
        }
    }

    Ok(ExportFunc {
        name,
        args,
        return_type,
    })
}

pub fn parse_string_field(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if after_colon.starts_with("null") {
        return None;
    }
    if !after_colon.starts_with('"') {
        return None;
    }
    let content = &after_colon[1..];
    let mut end = 0;
    let bytes = content.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'\\' {
            end += 2;
        } else if bytes[end] == b'"' {
            break;
        } else {
            end += 1;
        }
    }
    Some(content[..end].replace("\\\"", "\"").replace("\\\\", "\\"))
}

fn parse_nullable_string_field(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if after_colon.starts_with("null") {
        return None;
    }
    parse_string_field(json, key)
}

fn find_matching_bracket(s: &str) -> Result<usize, String> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }
    Err("unmatched '['".into())
}

fn find_matching_brace(s: &str) -> Result<usize, String> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }
    Err("unmatched '{'".into())
}

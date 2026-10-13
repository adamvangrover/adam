use ring::digest::{Context, SHA256};
use serde_json::Value;

/// Computes a canonical SHA-256 hash of a JSON value using sorted keys.
pub fn compute_canonical_hash(value: &Value) -> String {
    let canonical_str = to_canonical_json_string(value);
    let mut hasher = Context::new(&SHA256);
    hasher.update(canonical_str.as_bytes());
    hex::encode(hasher.finish().as_ref())
}

/// Recursively formats JSON value into deterministic canonical string representation.
pub fn to_canonical_json_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s)),
        Value::Array(arr) => {
            let elems: Vec<String> = arr.iter().map(to_canonical_json_string).collect();
            format!("[{}]", elems.join(","))
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let pairs: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_else(|_| format!("\"{}\"", k)),
                        to_canonical_json_string(&map[k])
                    )
                })
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_hash_consistency() {
        let v1 = serde_json::json!({ "b": 2, "a": 1 });
        let v2 = serde_json::json!({ "a": 1, "b": 2 });
        assert_eq!(compute_canonical_hash(&v1), compute_canonical_hash(&v2));
    }
}

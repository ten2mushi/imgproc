use crate::error::{Error, Result};
use std::collections::HashMap;

/// Interpolates variables in a string value.
///
/// Supports `${variable_name}` syntax. Variables must be defined in the variables map.
///
/// # Arguments
///
/// * `value` - The string value that may contain variable references
/// * `variables` - Map of variable names to their values
///
/// # Errors
///
/// Returns an error if a referenced variable is not found.
pub fn interpolate_string(
    value: &str,
    variables: &HashMap<String, serde_yaml::Value>,
) -> Result<String> {
    let mut result = value.to_string();
    let mut start = 0;

    while let Some(begin) = result[start..].find("${") {
        let begin = start + begin;
        if let Some(end) = result[begin..].find('}') {
            let end = begin + end;
            let var_name = &result[begin + 2..end];

            let var_value = variables
                .get(var_name)
                .ok_or_else(|| Error::VariableNotFound(var_name.to_string()))?;

            let replacement = yaml_value_to_string(var_value);
            result.replace_range(begin..=end, &replacement);
            start = begin + replacement.len();
        } else {
            break;
        }
    }

    Ok(result)
}

/// Interpolates variables in a YAML value.
///
/// Recursively processes maps and sequences, interpolating strings.
///
/// # Arguments
///
/// * `value` - The YAML value to interpolate
/// * `variables` - Map of variable names to their values
///
/// # Errors
///
/// Returns an error if a referenced variable is not found.
pub fn interpolate_value(
    value: &serde_yaml::Value,
    variables: &HashMap<String, serde_yaml::Value>,
) -> Result<serde_yaml::Value> {
    match value {
        serde_yaml::Value::String(s) => {
            let interpolated = interpolate_string(s, variables)?;
            // Try to parse the interpolated string back to a number if it was fully substituted
            // This handles cases like "${variable}" where the whole value is a variable reference
            if interpolated.chars().all(|c| c.is_ascii_digit() || c == '.') {
                // Try parsing as u64 first
                if let Ok(num) = interpolated.parse::<u64>() {
                    return Ok(serde_yaml::Value::Number(num.into()));
                }
                // Try parsing as f64
                if let Ok(num) = interpolated.parse::<f64>() {
                    return Ok(serde_yaml::Value::Number(
                        serde_yaml::Number::from(num)
                    ));
                }
            }
            Ok(serde_yaml::Value::String(interpolated))
        }
        serde_yaml::Value::Mapping(map) => {
            let mut new_map = serde_yaml::Mapping::new();
            for (k, v) in map {
                let new_v = interpolate_value(v, variables)?;
                new_map.insert(k.clone(), new_v);
            }
            Ok(serde_yaml::Value::Mapping(new_map))
        }
        serde_yaml::Value::Sequence(seq) => {
            let mut new_seq = Vec::new();
            for item in seq {
                new_seq.push(interpolate_value(item, variables)?);
            }
            Ok(serde_yaml::Value::Sequence(new_seq))
        }
        // Numbers, bools, and null pass through unchanged
        other => Ok(other.clone()),
    }
}

/// Converts a YAML value to its string representation.
fn yaml_value_to_string(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::Null => String::new(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Sequence(_) | serde_yaml::Value::Mapping(_) => {
            // For complex types, serialize to JSON
            serde_json::to_string(value).unwrap_or_default()
        }
        serde_yaml::Value::Tagged(tagged) => yaml_value_to_string(&tagged.value),
    }
}

/// Interpolates variables in a parameters map.
///
/// # Arguments
///
/// * `params` - The parameters map to interpolate
/// * `variables` - Map of variable names to their values
///
/// # Errors
///
/// Returns an error if a referenced variable is not found.
pub fn interpolate_params(
    params: &HashMap<String, serde_yaml::Value>,
    variables: &HashMap<String, serde_yaml::Value>,
) -> Result<HashMap<String, serde_yaml::Value>> {
    let mut result = HashMap::new();

    for (key, value) in params {
        let interpolated = interpolate_value(value, variables)?;
        result.insert(key.clone(), interpolated);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variables() -> HashMap<String, serde_yaml::Value> {
        let mut vars = HashMap::new();
        vars.insert(
            "size".to_string(),
            serde_yaml::Value::Number(800.into()),
        );
        vars.insert(
            "sigma".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(1.5)),
        );
        vars.insert(
            "format".to_string(),
            serde_yaml::Value::String("png".to_string()),
        );
        vars
    }

    #[test]
    fn test_interpolate_string_simple() {
        let vars = make_variables();
        let result = interpolate_string("${size}", &vars).unwrap();
        assert_eq!(result, "800");
    }

    #[test]
    fn test_interpolate_string_embedded() {
        let vars = make_variables();
        let result = interpolate_string("output_${size}.${format}", &vars).unwrap();
        assert_eq!(result, "output_800.png");
    }

    #[test]
    fn test_interpolate_string_multiple() {
        let vars = make_variables();
        let result = interpolate_string("${size}x${size}", &vars).unwrap();
        assert_eq!(result, "800x800");
    }

    #[test]
    fn test_interpolate_string_missing_variable() {
        let vars = make_variables();
        let result = interpolate_string("${missing}", &vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_value_number() {
        let vars = make_variables();
        let value = serde_yaml::Value::Number(42.into());
        let result = interpolate_value(&value, &vars).unwrap();
        assert_eq!(result, value);
    }

    #[test]
    fn test_interpolate_params() {
        let vars = make_variables();
        let mut params = HashMap::new();
        params.insert(
            "width".to_string(),
            serde_yaml::Value::String("${size}".to_string()),
        );
        params.insert(
            "height".to_string(),
            serde_yaml::Value::String("${size}".to_string()),
        );

        let result = interpolate_params(&params, &vars).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get("width").unwrap(),
            &serde_yaml::Value::String("800".to_string())
        );
    }
}

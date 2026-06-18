const TENSOR_RS: &str = include_str!("../src/tensor.rs");

#[test]
fn cuda_backed_tensor_methods_route_before_cpu_dispatch_or_materialization() {
    for case in [
        GuardCase {
            method: "fn binary_op(",
            cuda_route: "return self.cuda_binary_op",
            cpu_path: "dispatch::resolve_binary",
        },
        GuardCase {
            method: "pub fn matmul(",
            cuda_route: "return self.cuda_matmul",
            cpu_path: "dispatch::resolve_matmul",
        },
        GuardCase {
            method: "pub fn relu(",
            cuda_route: "return self.cuda_relu",
            cpu_path: "dispatch::resolve_unary",
        },
        GuardCase {
            method: "pub fn gelu(",
            cuda_route: "return self.cuda_gelu",
            cpu_path: "dispatch::resolve_unary",
        },
        GuardCase {
            method: "pub fn layer_norm_last_dim(",
            cuda_route: "return self.cuda_layer_norm_last_dim",
            cpu_path: "dispatch::resolve_layer_norm_last_dim",
        },
        GuardCase {
            method: "pub fn embedding(",
            cuda_route: "return self.cuda_embedding",
            cpu_path: "dispatch::resolve_embedding",
        },
        GuardCase {
            method: "pub fn causal_self_attention(",
            cuda_route: "return self.cuda_causal_self_attention",
            cpu_path: "dispatch::resolve_causal_self_attention",
        },
        GuardCase {
            method: "pub fn softmax_dim(",
            cuda_route: "return self.cuda_softmax_dim",
            cpu_path: "dispatch::resolve_unary",
        },
        GuardCase {
            method: "pub fn sum(",
            cuda_route: "return self.cuda_reduce_all",
            cpu_path: "dispatch::resolve_reduction",
        },
        GuardCase {
            method: "pub fn mean(",
            cuda_route: "return self.cuda_reduce_all",
            cpu_path: "dispatch::resolve_reduction",
        },
        GuardCase {
            method: "pub fn cross_entropy_for_logits(",
            cuda_route: "return self.cuda_cross_entropy_for_logits",
            cpu_path: "dispatch::resolve_unary",
        },
        GuardCase {
            method: "pub fn cross_entropy_for_logits_tensor(",
            cuda_route: "return self.cuda_cross_entropy_for_logits_tensor",
            cpu_path: ".data_i64()",
        },
    ] {
        assert_route_before_cpu_path(case);
    }
}

#[derive(Clone, Copy)]
struct GuardCase {
    method: &'static str,
    cuda_route: &'static str,
    cpu_path: &'static str,
}

fn assert_route_before_cpu_path(case: GuardCase) {
    let body = method_body(case.method);
    let route_index = body
        .find(case.cuda_route)
        .unwrap_or_else(|| panic!("{} missing {}", case.method, case.cuda_route));
    let cpu_index = body
        .find(case.cpu_path)
        .unwrap_or_else(|| panic!("{} missing {}", case.method, case.cpu_path));
    assert!(
        route_index < cpu_index,
        "{} must route CUDA tensors via {} before reaching CPU-only path {}",
        case.method,
        case.cuda_route,
        case.cpu_path
    );
}

fn method_body(marker: &str) -> &str {
    let start = TENSOR_RS
        .find(marker)
        .unwrap_or_else(|| panic!("missing method marker {marker}"));
    let after_start = start + marker.len();
    let end = ["\n    pub fn ", "\n    pub(crate) fn ", "\n    fn "]
        .iter()
        .filter_map(|next_marker| {
            TENSOR_RS[after_start..]
                .find(next_marker)
                .map(|offset| after_start + offset)
        })
        .min()
        .unwrap_or(TENSOR_RS.len());
    &TENSOR_RS[start..end]
}

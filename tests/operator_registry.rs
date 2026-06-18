use heirloom::{
    operator_catalog, AliasPolicy, AutogradPolicy, DType, DTypeRule, Operator, OperatorInfo,
    OperatorKind,
};

#[test]
fn public_operator_catalog_describes_the_builtin_runtime_surface() {
    let catalog = operator_catalog();
    assert_eq!(catalog.len(), 18);

    let add = catalog
        .iter()
        .find(|schema| schema.operator == Operator::Add)
        .expect("add schema");
    assert_eq!(add.name, "aten.add");
    assert_eq!(add.kind, OperatorKind::Binary);
    assert_eq!(add.dtype_rule, DTypeRule::ArithmeticPromotion);
    assert_eq!(add.alias_policy, AliasPolicy::FreshOutput);
    assert_eq!(add.autograd_policy, AutogradPolicy::FloatingInputs);
    assert!(add.returns_fresh_output());
    assert!(add.should_record_autograd(DType::F32, true));
    assert!(!add.should_record_autograd(DType::I64, true));
    assert!(!add.should_record_autograd(DType::F32, false));

    let matmul = catalog
        .iter()
        .find(|schema| schema.operator == Operator::Matmul)
        .expect("matmul schema");
    assert_eq!(matmul.kind, OperatorKind::MatrixMultiply);
    assert_eq!(matmul.dtype_rule, DTypeRule::FloatingPromotion);

    let sum_dim = catalog
        .iter()
        .find(|schema| schema.operator == Operator::SumDim)
        .expect("sum_dim schema");
    assert_eq!(sum_dim.kind, OperatorKind::ReductionDim);
    assert_eq!(sum_dim.dtype_rule, DTypeRule::SumReduction);

    let gelu = catalog
        .iter()
        .find(|schema| schema.operator == Operator::Gelu)
        .expect("gelu schema");
    assert_eq!(gelu.name, "aten.gelu");
    assert_eq!(gelu.kind, OperatorKind::Unary);
    assert_eq!(gelu.dtype_rule, DTypeRule::PreserveFloating);

    let layer_norm = catalog
        .iter()
        .find(|schema| schema.operator == Operator::LayerNormLastDim)
        .expect("layer_norm schema");
    assert_eq!(layer_norm.name, "aten.layer_norm.last_dim");
    assert_eq!(layer_norm.kind, OperatorKind::Normalization);
    assert_eq!(layer_norm.dtype_rule, DTypeRule::SameFloatingInputs);

    let embedding = catalog
        .iter()
        .find(|schema| schema.operator == Operator::Embedding)
        .expect("embedding schema");
    assert_eq!(embedding.name, "aten.embedding");
    assert_eq!(embedding.kind, OperatorKind::Indexing);
    assert_eq!(embedding.dtype_rule, DTypeRule::EmbeddingLookup);

    let masked_fill = catalog
        .iter()
        .find(|schema| schema.operator == Operator::MaskedFill)
        .expect("masked_fill schema");
    assert_eq!(masked_fill.name, "aten.masked_fill");
    assert_eq!(masked_fill.kind, OperatorKind::Masking);
    assert_eq!(masked_fill.dtype_rule, DTypeRule::MaskPreserve);

    let argmax = catalog
        .iter()
        .find(|schema| schema.operator == Operator::ArgmaxLastDim)
        .expect("argmax schema");
    assert_eq!(argmax.name, "aten.argmax.last_dim");
    assert_eq!(argmax.kind, OperatorKind::Selection);
    assert_eq!(argmax.dtype_rule, DTypeRule::ArgmaxIndex);
    assert_eq!(argmax.autograd_policy, AutogradPolicy::NonDifferentiable);
    assert!(!argmax.should_record_autograd(DType::I64, true));

    let attention = catalog
        .iter()
        .find(|schema| schema.operator == Operator::CausalSelfAttention)
        .expect("causal attention schema");
    assert_eq!(attention.name, "heirloom.causal_self_attention");
    assert_eq!(attention.kind, OperatorKind::Attention);
    assert_eq!(attention.dtype_rule, DTypeRule::SameFloatingInputs);
}

#[test]
fn catalog_distinguishes_aten_like_ops_from_heirloom_specific_fused_ops() {
    let catalog = operator_catalog();
    assert!(catalog
        .iter()
        .any(|schema| schema.name == "heirloom.cross_entropy_for_logits"));
    assert!(catalog
        .iter()
        .any(|schema| schema.name == "heirloom.causal_self_attention"));
    assert!(
        catalog
            .iter()
            .filter(|schema| schema.name.starts_with("aten."))
            .count()
            >= 16
    );
}

#[test]
fn operator_autograd_policy_is_executable_metadata() {
    let nondifferentiable = OperatorInfo {
        operator: Operator::Relu,
        name: "test.non_differentiable",
        kind: OperatorKind::Unary,
        dtype_rule: DTypeRule::PreserveFloating,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::NonDifferentiable,
    };

    assert!(nondifferentiable.returns_fresh_output());
    assert!(!nondifferentiable.should_record_autograd(DType::F32, true));
}

load(
    "@org_tensorflow//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "build_config.bzl",
    "gcp_monitoring_grpc_deps",
    "gcp_monitoring_proto_deps",
    "gcp_monitoring_proto_header_deps",
)

cc_library(
    name = "stackdriver_client",
    srcs = [
        "stackdriver_client.cc",
    ],
    hdrs = [
        "stackdriver_client.h",
    ],
    deps = [
        "@com_google_protobuf//:protobuf_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
    ] + gcp_monitoring_grpc_deps() + gcp_monitoring_proto_header_deps(),
)

cc_test(
    name = "stackdriver_client_test",
    srcs = [
        "stackdriver_client_test.cc",
    ],
    deps = [
        ":stackdriver_client",
        "@com_google_googletest//:gtest_main",
    ] + gcp_monitoring_grpc_deps() + gcp_monitoring_proto_deps(),
)

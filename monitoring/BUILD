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
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf_headers",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_util",
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

cc_library(
    name = "stackdriver_config",
    srcs = [
        "stackdriver_config.cc",
    ],
    hdrs = [
        "stackdriver_config.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "stackdriver_exporter",
    srcs = [
        "stackdriver_exporter.cc",
    ],
    hdrs = [
        "stackdriver_exporter.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":stackdriver_client",
        ":stackdriver_config",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
    ],
)

cc_library(
    name = "stackdriver_exporter_protos",
    visibility = [
        "//visibility:public",
    ],
    deps = gcp_monitoring_proto_deps(),
)

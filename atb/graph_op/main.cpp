#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/types.h"
#include <atb/atb_infer.h>
#include <atb/utils.h>
#include <atb/infer_op_params.h>

#include "utils.h"

const int32_t DEVICE_ID = 0;
const uint32_t X_DIM_0 = 2;
const uint32_t X_DIM_1 = 3;
const uint32_t WEIGHT_DIM_0 = 3;
const uint32_t WEIGHT_DIM_1 = 2;
const uint32_t BIAS_DIM_0 = 2;

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @return atb::SVector<atb::Tensor> atb::VariantPack
 * @note 需要传入所有host侧tensor
 */
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream)
{
    // 创建shape为[2, 3]的输入a tensor
    atb::Tensor a = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>{1, 2, 3, 4, 5, 6},
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {X_DIM_0, X_DIM_1});

    // 创建shape为[2, 3]的输入b tensor
    atb::Tensor b = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>{1, 2, 3, 4, 5, 6},
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {X_DIM_0, X_DIM_1});

    // 创建shape为[2, 3]的输入c tensor
    atb::Tensor c = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>{1, 2, 3, 4, 5, 6},
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {X_DIM_0, X_DIM_1});

    // 创建shape为[2, 3]的输入d tensor
    atb::Tensor d = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>{1, 2, 3, 4, 5, 6},
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {X_DIM_0, X_DIM_1});

    atb::SVector<atb::Tensor> inTensors = {a, b, c, d};
    return inTensors;
}

atb::Status CreateGraphOperation(atb::Operation **operation)
{
    // 构图流程
    // 图算子的输入a,b,c,d
    // 计算公式：(a+b) + (c+d)
    // 输入是4个参数，输出是1个参数，有3个add算子，中间产生的临时输出是2个
    atb::GraphParam opGraph;

    opGraph.inTensorNum = 4;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(3);

    enum InTensorId
    { // 定义各TensorID
        IN_TENSOR_A = 0,
        IN_TENSOR_B,
        IN_TENSOR_C,
        IN_TENSOR_D,
        ADD3_OUT,
        ADD1_OUT,
        ADD2_OUT
    };

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode3 = opGraph.nodes.at(nodeId++);

    atb::Operation *op = nullptr;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    auto status = atb::CreateOperation(addParam, &addNode.operation);
    CHECK_RET(status, "addParam CreateOperation failed. status: " + std::to_string(status));
    addNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    addNode.outTensorIds = {ADD1_OUT};

    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    status = atb::CreateOperation(addParam2, &addNode2.operation);
    CHECK_RET(status, "addParam2 CreateOperation failed. status: " + std::to_string(status));
    addNode2.inTensorIds = {IN_TENSOR_C, IN_TENSOR_D};
    addNode2.outTensorIds = {ADD2_OUT};

    atb::infer::ElewiseParam addParam3;
    addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    status = atb::CreateOperation(addParam3, &addNode3.operation);
    CHECK_RET(status, "addParam3 CreateOperation failed. status: " + std::to_string(status));
    addNode3.inTensorIds = {ADD1_OUT, ADD2_OUT};
    addNode3.outTensorIds = {ADD3_OUT};

    status = atb::CreateOperation(opGraph, operation);
    CHECK_RET(status, "GraphParam CreateOperation failed. status: " + std::to_string(status));

    return atb::NO_ERROR;
}

int main(int argc, char **argv)
{
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // 创建op
    atb::Operation *graphOp = nullptr;
    CHECK_STATUS(CreateGraphOperation(&graphOp));

    // 准备输入tensor
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream);  // 放入输入tensor

    // 准备输出tensor
    atb::Tensor output = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {X_DIM_0, X_DIM_1});
    variantPack.outTensors = {output};  // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(graphOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // graphOp执行
    graphOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream));  // 流同步，等待device侧任务计算完成

    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(graphOp));  // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context));  // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "graph op demo success!" << std::endl;
    return 0;
}

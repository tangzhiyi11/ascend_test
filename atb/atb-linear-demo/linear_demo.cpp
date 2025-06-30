#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/types.h"
#include <atb/atb_infer.h>

#include "demo_util.h"
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
    // 创建shape为[2, 3]的输入x tensor
    atb::Tensor xFloat = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>{1, 2, 3, 4, 5, 6},
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {X_DIM_0, X_DIM_1});
    // 创建shape为[3, 2]的输入weight tensor
    atb::Tensor weightFloat = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>{1, 2, 3, 4, 5, 6},
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {WEIGHT_DIM_0, WEIGHT_DIM_1});
    // 创建shape为[2]的输入bias tensor
    atb::Tensor biasFloat = CreateTensorFromVector(
        contextPtr, stream, std::vector<float>(BIAS_DIM_0, 1.0), ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {1, BIAS_DIM_0});
    atb::SVector<atb::Tensor> inTensors = {xFloat, weightFloat, biasFloat};
    return inTensors;
}

/**
 * @brief 创建一个linear operation
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *CreateLinearOperation()
{
    atb::infer::LinearParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.hasBias = true;
    param.outDataType = aclDataType::ACL_DT_UNDEFINED;
    param.enAccum = false;
    param.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
    atb::Operation *LinearOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &LinearOp));
    return LinearOp;
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
    atb::Operation *linearOp = CreateLinearOperation();
    // 准备输入tensor
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream);  // 放入输入tensor
    // 准备输出tensor
    atb::Tensor output = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {X_DIM_0, WEIGHT_DIM_1});
    variantPack.outTensors = {output};  // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(linearOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // linear执行
    linearOp->Execute(variantPack, workspacePtr, workspaceSize, context);
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
    CHECK_STATUS(atb::DestroyOperation(linearOp));  // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context));  // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "Linear demo success!" << std::endl;
    return 0;
}

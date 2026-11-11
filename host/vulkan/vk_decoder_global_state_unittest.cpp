// copyright (c) 2022 the android open source project
//
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
// http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// limitations under the license.

#include "vk_decoder_global_state.cpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <optional>
#include <thread>

#include "gfxstream/testing/TestUtils.h"

namespace gfxstream {
namespace host {
namespace vk {

// Handles the device-cleanup test needs to inspect after teardown.
struct CommandBufferCleanupProbe {
    VkCommandBuffer boxedCmdBuf = VK_NULL_HANDLE;
    std::shared_ptr<OrderMaintenanceInfo> cmdBufOrder;
};

struct VkDecoderGlobalStateTestPeer {
    VkDecoderGlobalState::Impl impl;

    void registerDescriptorPool(VkDescriptorPool pool, std::vector<uint64_t> poolIds = {}) {
        std::lock_guard<std::mutex> lock(impl.mMutex);
        DescriptorPoolInfo info;
        info.preallocatedBoxedSets = std::move(poolIds);
        impl.mDescriptorPoolInfo[pool] = std::move(info);
    }

    void collectDescriptorPoolIds(gfxstream::base::BumpPool* pool, VkDevice device,
                                  VkDescriptorPool descriptorPool, uint32_t* pPoolIdCount,
                                  uint64_t* pPoolIds) {
        impl.on_vkCollectDescriptorPoolIdsGOOGLE(pool, kInvalidSnapshotApiCallHandle, device,
                                                 descriptorPool, pPoolIdCount, pPoolIds);
    }

    VkDescriptorSet getOrAllocateDescriptorSetFromPoolAndId(
        VulkanDispatch* vk, VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout setLayout,
        uint64_t poolId, uint32_t pendingAlloc, bool* didAlloc) {
        std::lock_guard<std::mutex> lock(impl.mMutex);
        return impl.getOrAllocateDescriptorSetFromPoolAndIdLocked(
            vk, device, pool, setLayout, poolId, pendingAlloc, didAlloc);
    }

    void commitDescriptorSetUpdates(
        gfxstream::base::BumpPool* pool, VulkanDispatch* vk, VkDevice device,
        uint32_t descriptorPoolCount, const VkDescriptorPool* pDescriptorPools,
        uint32_t descriptorSetCount, const VkDescriptorSetLayout* pDescriptorSetLayouts,
        const uint64_t* pDescriptorSetPoolIds, const uint32_t* pDescriptorSetWhichPool,
        const uint32_t* pDescriptorSetPendingAllocation,
        const uint32_t* pDescriptorWriteStartingIndices, uint32_t pendingDescriptorWriteCount,
        const VkWriteDescriptorSet* pPendingDescriptorWrites) {
        std::lock_guard<std::mutex> lock(impl.mMutex);
        impl.on_vkQueueCommitDescriptorSetUpdatesGOOGLELocked(
            pool, kInvalidSnapshotApiCallHandle, vk, device, descriptorPoolCount, pDescriptorPools,
            descriptorSetCount, pDescriptorSetLayouts, pDescriptorSetPoolIds,
            pDescriptorSetWhichPool, pDescriptorSetPendingAllocation,
            pDescriptorWriteStartingIndices, pendingDescriptorWriteCount, pPendingDescriptorWrites);
    }

    void destroyCommandPoolWithExclusiveInfo(
        VkDevice device, VulkanDispatch* deviceDispatch, VkCommandPool commandPool,
        CommandPoolInfo& commandPoolInfo,
        std::unordered_map<VkCommandBuffer, CommandBufferInfo>& commandBufferInfos) {
        impl.destroyCommandPoolWithExclusiveInfo(device, deviceDispatch, commandPool,
                                                 commandPoolInfo, commandBufferInfos);
    }

    // Boxes one object of each device-owned type and registers it in the Impl
    // info maps, as if the guest had created it on `unboxedDevice`.
    CommandBufferCleanupProbe registerDeviceObjectsForCleanupTest(VulkanDispatch* globalVk,
                                                                  VulkanDispatch* deviceVk,
                                                                  VkDevice unboxedDevice) {
        impl.m_vk = globalVk;

        VkDevice boxedDevice = new_boxed_VkDevice(unboxedDevice, deviceVk);
        VkCommandPool unboxedCmdPool = reinterpret_cast<VkCommandPool>(0xa001'0000ULL);
        VkCommandPool boxedCmdPool = new_boxed_non_dispatchable_VkCommandPool(unboxedCmdPool);
        VkCommandBuffer unboxedCmdBuf = reinterpret_cast<VkCommandBuffer>(0xa002'0000ULL);
        VkCommandBuffer boxedCmdBuf = new_boxed_VkCommandBuffer(unboxedCmdBuf, deviceVk);
        BoxedHandleInfo* cmdBufBoxedInfo =
            sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxedCmdBuf));
        EXPECT_NE(nullptr, cmdBufBoxedInfo);
        if (cmdBufBoxedInfo) {
            cmdBufBoxedInfo->readStream =
                std::make_unique<VulkanMemReadingStream>(nullptr, gfxstream::host::FeatureSet{});
        }
        CommandBufferCleanupProbe probe;
        probe.boxedCmdBuf = boxedCmdBuf;
        probe.cmdBufOrder = ordmaint_VkCommandBuffer(boxedCmdBuf);

        VkBuffer unboxedBuffer = reinterpret_cast<VkBuffer>(0xa003'0000ULL);
        VkBuffer boxedBuffer = new_boxed_non_dispatchable_VkBuffer(unboxedBuffer);
        VkDeviceMemory unboxedMem = reinterpret_cast<VkDeviceMemory>(0xa004'0000ULL);
        VkDeviceMemory boxedMem = new_boxed_non_dispatchable_VkDeviceMemory(unboxedMem);
        VkDescriptorPool unboxedDescPool = reinterpret_cast<VkDescriptorPool>(0xa005'0000ULL);
        VkDescriptorPool boxedDescPool =
            new_boxed_non_dispatchable_VkDescriptorPool(unboxedDescPool);
        VkDescriptorUpdateTemplate unboxedTemplate =
            reinterpret_cast<VkDescriptorUpdateTemplate>(0xa006'0000ULL);
        VkDescriptorUpdateTemplate boxedTemplate =
            new_boxed_non_dispatchable_VkDescriptorUpdateTemplate(unboxedTemplate);
        VkSamplerYcbcrConversion unboxedYcbcr =
            reinterpret_cast<VkSamplerYcbcrConversion>(0xa007'0000ULL);
        VkSamplerYcbcrConversion boxedYcbcr =
            new_boxed_non_dispatchable_VkSamplerYcbcrConversion(unboxedYcbcr);
        VkShaderModule unboxedShader = reinterpret_cast<VkShaderModule>(0xa008'0000ULL);
        VkShaderModule boxedShader = new_boxed_non_dispatchable_VkShaderModule(unboxedShader);
        VkPipeline unboxedPipeline = reinterpret_cast<VkPipeline>(0xa009'0000ULL);
        VkPipeline boxedPipeline = new_boxed_non_dispatchable_VkPipeline(unboxedPipeline);
        VkPipelineCache unboxedCache = reinterpret_cast<VkPipelineCache>(0xa00a'0000ULL);
        VkPipelineCache boxedCache = new_boxed_non_dispatchable_VkPipelineCache(unboxedCache);
        VkPipelineLayout unboxedLayout = reinterpret_cast<VkPipelineLayout>(0xa00b'0000ULL);
        VkPipelineLayout boxedLayout = new_boxed_non_dispatchable_VkPipelineLayout(unboxedLayout);
        VkFramebuffer unboxedFb = reinterpret_cast<VkFramebuffer>(0xa00c'0000ULL);
        VkFramebuffer boxedFb = new_boxed_non_dispatchable_VkFramebuffer(unboxedFb);
        VkRenderPass unboxedRp = reinterpret_cast<VkRenderPass>(0xa00d'0000ULL);
        VkRenderPass boxedRp = new_boxed_non_dispatchable_VkRenderPass(unboxedRp);

        {
            std::lock_guard<std::mutex> lock(impl.mMutex);
            DeviceInfo& devInfo = impl.mDeviceInfo[unboxedDevice];
            devInfo.boxed = boxedDevice;
            devInfo.decompPipelines =
                std::make_unique<GpuDecompressionPipelineManager>(deviceVk, unboxedDevice);
            devInfo.deviceOpTracker = std::make_shared<DeviceOpTracker>(unboxedDevice, deviceVk);
            devInfo.externalFencePool =
                std::make_unique<ExternalFencePool<VulkanDispatch>>(deviceVk, unboxedDevice);

            CommandPoolInfo& poolInfo = impl.mCommandPoolInfo[unboxedCmdPool];
            poolInfo.device = unboxedDevice;
            poolInfo.boxed = boxedCmdPool;
            poolInfo.cmdBuffers.insert(unboxedCmdBuf);

            CommandBufferInfo& cmdInfo = impl.mCommandBufferInfo[unboxedCmdBuf];
            cmdInfo.device = unboxedDevice;
            cmdInfo.cmdPool = unboxedCmdPool;
            cmdInfo.boxed = boxedCmdBuf;

            BufferInfo& bufInfo = impl.mBufferInfo[unboxedBuffer];
            bufInfo.device = unboxedDevice;
            bufInfo.boxed = boxedBuffer;

            MemoryInfo& memInfo = impl.mMemoryInfo[unboxedMem];
            memInfo.device = unboxedDevice;
            memInfo.boxed = boxedMem;
            memInfo.directMapped = true;

            DescriptorPoolInfo& descPoolInfo = impl.mDescriptorPoolInfo[unboxedDescPool];
            descPoolInfo.device = unboxedDevice;
            descPoolInfo.boxed = boxedDescPool;

            DescriptorUpdateTemplateInfo& tmplInfo =
                impl.mDescriptorUpdateTemplateInfo[unboxedTemplate];
            tmplInfo.device = unboxedDevice;
            tmplInfo.boxed = boxedTemplate;
            tmplInfo.data.resize(256, 0xab);

            SamplerYcbcrConversionInfo& ycbcrInfo = impl.mSamplerYcbcrConversionInfo[unboxedYcbcr];
            ycbcrInfo.device = unboxedDevice;
            ycbcrInfo.boxed = boxedYcbcr;

            ShaderModuleInfo& shaderInfo = impl.mShaderModuleInfo[unboxedShader];
            shaderInfo.device = unboxedDevice;
            shaderInfo.boxed = boxedShader;

            PipelineInfo& pipeInfo = impl.mPipelineInfo[unboxedPipeline];
            pipeInfo.device = unboxedDevice;
            pipeInfo.boxed = boxedPipeline;

            PipelineCacheInfo& cacheInfo = impl.mPipelineCacheInfo[unboxedCache];
            cacheInfo.device = unboxedDevice;
            cacheInfo.boxed = boxedCache;

            PipelineLayoutInfo& layoutInfo = impl.mPipelineLayoutInfo[unboxedLayout];
            layoutInfo.device = unboxedDevice;
            layoutInfo.boxed = boxedLayout;

            FramebufferInfo& fbInfo = impl.mFramebufferInfo[unboxedFb];
            fbInfo.device = unboxedDevice;
            fbInfo.boxed = boxedFb;

            RenderPassInfo& rpInfo = impl.mRenderPassInfo[unboxedRp];
            rpInfo.device = unboxedDevice;
            rpInfo.boxed = boxedRp;
        }
        return probe;
    }

    // Mirrors process cleanup: extracts the device and every object it owns
    // from the Impl info maps.
    InstanceObjects::DeviceObjects extractDeviceObjectsForCleanupTest(VkDevice unboxedDevice) {
        InstanceObjects::DeviceObjects deviceObjects;
        std::lock_guard<std::mutex> lock(impl.mMutex);
        auto devIt = impl.mDeviceInfo.find(unboxedDevice);
        deviceObjects.device = impl.mDeviceInfo.extract(devIt);
        impl.extractDeviceAndDependenciesLocked(unboxedDevice, deviceObjects);
        return deviceObjects;
    }

    size_t descriptorUpdateTemplateInfoCount() {
        std::lock_guard<std::mutex> lock(impl.mMutex);
        return impl.mDescriptorUpdateTemplateInfo.size();
    }

    size_t samplerYcbcrConversionInfoCount() {
        std::lock_guard<std::mutex> lock(impl.mMutex);
        return impl.mSamplerYcbcrConversionInfo.size();
    }

    void destroyDeviceObjects(InstanceObjects::DeviceObjects& deviceObjects) {
        impl.destroyDeviceObjects(deviceObjects);
    }

    // Registers `unboxedInstance` with already boxed physical devices, as if
    // the first vkEnumeratePhysicalDevices call had populated them. Returns
    // the boxed instance handle.
    VkInstance registerInstanceWithPhysicalDevicesForCleanupTest(
        VulkanDispatch* globalVk, VulkanDispatch* instanceVk, VkInstance unboxedInstance,
        const std::vector<VkPhysicalDevice>& unboxedPhysicalDevices,
        std::vector<VkPhysicalDevice> boxedPhysicalDevices) {
        impl.m_vk = globalVk;
        VkInstance boxedInstance = new_boxed_VkInstance(unboxedInstance, instanceVk);

        std::lock_guard<std::mutex> lock(impl.mMutex);
        InstanceInfo& instanceInfo = impl.mInstanceInfo[unboxedInstance];
        instanceInfo.boxed = boxedInstance;
        instanceInfo.boxedPhysicalDevices = std::move(boxedPhysicalDevices);
        for (VkPhysicalDevice unboxedPhysicalDevice : unboxedPhysicalDevices) {
            impl.mPhysdevInfo[unboxedPhysicalDevice].instance = unboxedInstance;
        }
        return boxedInstance;
    }

    VkResult enumeratePhysicalDevices(VkInstance boxedInstance, uint32_t* pPhysicalDeviceCount,
                                      VkPhysicalDevice* pPhysicalDevices) {
        return impl.on_vkEnumeratePhysicalDevices(nullptr, kInvalidSnapshotApiCallHandle,
                                                  boxedInstance, pPhysicalDeviceCount,
                                                  pPhysicalDevices);
    }

    size_t physicalDeviceInfoCount() {
        std::lock_guard<std::mutex> lock(impl.mMutex);
        return impl.mPhysdevInfo.size();
    }

    // Mirrors vkDestroyInstanceImpl: extracts the instance and everything it
    // owns from the Impl info maps.
    InstanceObjects extractInstanceObjectsForCleanupTest(VkInstance unboxedInstance) {
        InstanceObjects instanceObjects;
        std::lock_guard<std::mutex> lock(impl.mMutex);
        impl.extractInstanceAndDependenciesLocked(unboxedInstance, instanceObjects);
        return instanceObjects;
    }

    void destroyInstanceObjects(InstanceObjects& instanceObjects) {
        impl.destroyInstanceObjects(instanceObjects);
    }

    void setSnapshotsEnabled(bool enabled) { impl.mSnapshotsEnabled = enabled; }

    VkDevice registerSnapshotImageTestRig(VulkanDispatch* globalVk, VulkanDispatch* instanceVk,
                                          VulkanDispatch* deviceVk, VkInstance unboxedInstance,
                                          VkPhysicalDevice unboxedPhysicalDevice,
                                          VkDevice unboxedDevice, uint32_t apiVersion,
                                          std::vector<std::string> instanceExtensions = {}) {
        impl.m_vk = globalVk;
        VkInstance boxedInstance = new_boxed_VkInstance(unboxedInstance, instanceVk);
        VkPhysicalDevice boxedPhysDev =
            new_boxed_VkPhysicalDevice(unboxedPhysicalDevice, instanceVk);
        VkDevice boxedDevice = new_boxed_VkDevice(unboxedDevice, deviceVk);

        VkPhysicalDeviceMemoryProperties hostMemProps = {};
        hostMemProps.memoryHeapCount = 1;
        hostMemProps.memoryHeaps[0].size = 64ULL * 1024 * 1024;
        hostMemProps.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
        hostMemProps.memoryTypeCount = 1;
        hostMemProps.memoryTypes[0].heapIndex = 0;
        hostMemProps.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        std::lock_guard<std::mutex> lock(impl.mMutex);
        InstanceInfo& instanceInfo = impl.mInstanceInfo[unboxedInstance];
        instanceInfo.apiVersion = apiVersion;
        instanceInfo.enabledExtensionNames = std::move(instanceExtensions);
        instanceInfo.boxed = boxedInstance;
        instanceInfo.boxedPhysicalDevices = {boxedPhysDev};

        PhysicalDeviceInfo& physDevInfo = impl.mPhysdevInfo[unboxedPhysicalDevice];
        physDevInfo.instance = unboxedInstance;
        physDevInfo.props.apiVersion = apiVersion;
        physDevInfo.memoryPropertiesHelper = std::make_unique<EmulatedPhysicalDeviceMemoryProperties>(
            hostMemProps, /*hostColorBufferMemoryTypeIndex=*/0, gfxstream::host::FeatureSet{});

        DeviceInfo& devInfo = impl.mDeviceInfo[unboxedDevice];
        devInfo.physicalDevice = unboxedPhysicalDevice;
        devInfo.boxed = boxedDevice;
        devInfo.maxImageDimension2D = 4096;
        devInfo.decompPipelines =
            std::make_unique<GpuDecompressionPipelineManager>(deviceVk, unboxedDevice);
        devInfo.deviceOpTracker = std::make_shared<DeviceOpTracker>(unboxedDevice, deviceVk);
        devInfo.externalFencePool =
            std::make_unique<ExternalFencePool<VulkanDispatch>>(deviceVk, unboxedDevice);
        return boxedDevice;
    }

    VkResult createImage(gfxstream::base::BumpPool* pool, VkDevice boxedDevice,
                         const VkImageCreateInfo* pCreateInfo, VkImage* pImage) {
        return impl.on_vkCreateImage(pool, kInvalidSnapshotApiCallHandle, boxedDevice, pCreateInfo,
                                     nullptr, pImage);
    }

    void getDeviceImageMemoryRequirements(gfxstream::base::BumpPool* pool, VkDevice boxedDevice,
                                          const VkDeviceImageMemoryRequirements* pInfo,
                                          VkMemoryRequirements2* pMemoryRequirements) {
        impl.on_vkGetDeviceImageMemoryRequirements(pool, kInvalidSnapshotApiCallHandle, boxedDevice,
                                                   pInfo, pMemoryRequirements);
    }
};

namespace {

TEST(VkDecoderGlobalStateDescriptorSetPoolIdTest, BogusPoolIdReturnsNullWithoutCrash) {
    VkDecoderGlobalStateTestPeer peer;
    VulkanDispatch vk = {};
    VkDevice device = reinterpret_cast<VkDevice>(0x1111'0000);
    VkDescriptorPool pool = reinterpret_cast<VkDescriptorPool>(0x2222'0000);
    VkDescriptorSetLayout setLayout = reinterpret_cast<VkDescriptorSetLayout>(0x3333'0000);

    peer.registerDescriptorPool(pool);

    bool didAlloc = true;
    constexpr uint64_t kBogusPoolId = 0xdeadbeef'cafebabeULL;
    EXPECT_EQ(nullptr, sBoxedHandleManager.get(kBogusPoolId));

    VkDescriptorSet result = peer.getOrAllocateDescriptorSetFromPoolAndId(
        &vk, device, pool, setLayout, kBogusPoolId, /*pendingAlloc=*/1, &didAlloc);
    EXPECT_EQ(VK_NULL_HANDLE, result);
    EXPECT_FALSE(didAlloc);

    gfxstream::base::BumpPool bumpPool;
    uint32_t whichPool = 0;
    uint32_t pendingAlloc = 1;
    uint32_t writeStartIndex = 0;
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    };
    peer.commitDescriptorSetUpdates(
        &bumpPool, &vk, device,
        /*descriptorPoolCount=*/1, &pool,
        /*descriptorSetCount=*/1, &setLayout, &kBogusPoolId, &whichPool, &pendingAlloc,
        &writeStartIndex, /*pendingDescriptorWriteCount=*/1, &write);
}

TEST(VkDecoderGlobalStateDescriptorSetPoolIdTest, UnallocatedAndAllocatedPoolIdHandling) {
    VkDecoderGlobalStateTestPeer peer;
    VulkanDispatch vk = {};
    VkDevice device = reinterpret_cast<VkDevice>(0x1111'0000);
    VkDescriptorPool pool = reinterpret_cast<VkDescriptorPool>(0x2222'0000);
    VkDescriptorSetLayout setLayout = reinterpret_cast<VkDescriptorSetLayout>(0x3333'0000);
    VkDescriptorSet expectedSet = reinterpret_cast<VkDescriptorSet>(0x4444'0000);

    peer.registerDescriptorPool(pool);

    VkDescriptorSet boxedNullSet = new_boxed_non_dispatchable_VkDescriptorSet(VK_NULL_HANDLE);
    uint64_t nullPoolId = reinterpret_cast<uint64_t>(boxedNullSet);
    bool didAlloc = true;
    EXPECT_EQ(VK_NULL_HANDLE,
              peer.getOrAllocateDescriptorSetFromPoolAndId(
                  &vk, device, pool, setLayout, nullPoolId, /*pendingAlloc=*/0, &didAlloc));
    EXPECT_FALSE(didAlloc);
    delete_VkDescriptorSet(boxedNullSet);

    VkDescriptorSet boxedValidSet = new_boxed_non_dispatchable_VkDescriptorSet(expectedSet);
    uint64_t validPoolId = reinterpret_cast<uint64_t>(boxedValidSet);
    didAlloc = true;
    EXPECT_EQ(expectedSet,
              peer.getOrAllocateDescriptorSetFromPoolAndId(
                  &vk, device, pool, setLayout, validPoolId, /*pendingAlloc=*/0, &didAlloc));
    EXPECT_FALSE(didAlloc);
    delete_VkDescriptorSet(boxedValidSet);
}

TEST(VkDecoderGlobalStateDescriptorSetPoolIdTest, CollectDescriptorPoolIdsClampsCount) {
    VkDecoderGlobalStateTestPeer peer;
    gfxstream::base::BumpPool bumpPool;
    VkDevice device = reinterpret_cast<VkDevice>(0x1111'0000);
    VkDescriptorPool pool = reinterpret_cast<VkDescriptorPool>(0x2222'0000);

    peer.registerDescriptorPool(pool, {101, 102, 103, 104});

    uint32_t count = 0;
    peer.collectDescriptorPoolIds(&bumpPool, device, pool, &count, nullptr);
    EXPECT_EQ(4u, count);

    uint64_t outIds[4] = {0, 0, 0, 0};
    count = 1;
    peer.collectDescriptorPoolIds(&bumpPool, device, pool, &count, outIds);
    EXPECT_EQ(1u, count);
    EXPECT_EQ(101u, outIds[0]);
    EXPECT_EQ(0u, outIds[1]);

    count = 10;
    peer.collectDescriptorPoolIds(&bumpPool, device, pool, &count, outIds);
    EXPECT_EQ(4u, count);
    EXPECT_EQ(101u, outIds[0]);
    EXPECT_EQ(102u, outIds[1]);
    EXPECT_EQ(103u, outIds[2]);
    EXPECT_EQ(104u, outIds[3]);
}

TEST(VkDecoderGlobalStateBoxedHandleTest, OrderMaintenanceInfoOutlivesDeletedHandle) {
    VkCommandBuffer boxed =
        new_boxed_VkCommandBuffer(reinterpret_cast<VkCommandBuffer>(0x5555'0000), nullptr);
    std::shared_ptr<OrderMaintenanceInfo> order = ordmaint_VkCommandBuffer(boxed);
    ASSERT_NE(nullptr, order);
    EXPECT_EQ(order, ordmaint_VkCommandBuffer(boxed));

    delete_VkCommandBuffer(boxed);
    EXPECT_EQ(nullptr, ordmaint_VkCommandBuffer(boxed));
    EXPECT_EQ(1, order.use_count());
}

TEST(VkDecoderGlobalStateBoxedHandleTest, ConcurrentOrderMaintenanceInfoLookupAndDelete) {
    constexpr int kIterations = 20000;
    std::atomic<uint64_t> currentHandle{0};
    std::atomic<bool> done{false};

    std::thread reader([&]() {
        while (!done.load(std::memory_order_acquire)) {
            VkCommandBuffer boxed =
                reinterpret_cast<VkCommandBuffer>(currentHandle.load(std::memory_order_acquire));
            std::shared_ptr<OrderMaintenanceInfo> order = ordmaint_VkCommandBuffer(boxed);
            if (order) {
                EXPECT_EQ(0u, __atomic_load_n(&order->sequenceNumber, __ATOMIC_ACQUIRE));
            }
        }
    });

    for (int i = 0; i < kIterations; ++i) {
        VkCommandBuffer boxed =
            new_boxed_VkCommandBuffer(reinterpret_cast<VkCommandBuffer>(0x5555'0000), nullptr);
        currentHandle.store(reinterpret_cast<uint64_t>(boxed), std::memory_order_release);
        delete_VkCommandBuffer(boxed);
    }

    done.store(true, std::memory_order_release);
    reader.join();
}

// Regression test for b/573373414: the host driver can hand out the same unboxed handle value
// for a new object right after the previous object was destroyed, and the new boxed handle may be
// registered before the old boxed handle is removed. Removing the old boxed handle must not drop
// the reverse mapping owned by the new boxed handle, otherwise unboxed_to_boxed_* lookups fail
// and (among other things) snapshot dependency tracking for dedicated allocations breaks.
TEST(BoxedHandleManagerReverseMapTest, RemoveOldBoxedHandleKeepsReusedUnboxedMapping) {
    VkImage underlying = reinterpret_cast<VkImage>(0x5555'0000);

    VkImage boxedOld = new_boxed_non_dispatchable_VkImage(underlying);
    ASSERT_NE(VK_NULL_HANDLE, boxedOld);
    EXPECT_EQ(boxedOld, unboxed_to_boxed_non_dispatchable_VkImage(underlying));

    // Host destroys the old image and immediately reuses the handle value for a new image, whose
    // boxed handle gets registered before the old boxed handle is deleted.
    VkImage boxedNew = new_boxed_non_dispatchable_VkImage(underlying);
    ASSERT_NE(VK_NULL_HANDLE, boxedNew);
    ASSERT_NE(boxedOld, boxedNew);
    EXPECT_EQ(boxedNew, unboxed_to_boxed_non_dispatchable_VkImage(underlying));

    delete_VkImage(boxedOld);
    EXPECT_EQ(nullptr, sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxedOld)));
    // The new boxed handle must still be reachable from the (reused) unboxed handle.
    EXPECT_EQ(boxedNew, unboxed_to_boxed_non_dispatchable_VkImage(underlying));

    delete_VkImage(boxedNew);
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(underlying));
}

TEST(BoxedHandleManagerReverseMapTest, UpdateOldUnboxedDoesNotDropReusedMapping) {
    VkImage underlyingA = reinterpret_cast<VkImage>(0x6666'0000);
    VkImage underlyingB = reinterpret_cast<VkImage>(0x7777'0000);

    VkImage boxed1 = new_boxed_non_dispatchable_VkImage(underlyingA);
    // Another boxed handle takes over underlyingA (reuse) before boxed1 is re-pointed.
    VkImage boxed2 = new_boxed_non_dispatchable_VkImage(underlyingA);
    EXPECT_EQ(boxed2, unboxed_to_boxed_non_dispatchable_VkImage(underlyingA));

    set_boxed_non_dispatchable_VkImage(boxed1, underlyingB);
    EXPECT_EQ(boxed1, unboxed_to_boxed_non_dispatchable_VkImage(underlyingB));
    // boxed2's mapping for underlyingA must survive boxed1's update.
    EXPECT_EQ(boxed2, unboxed_to_boxed_non_dispatchable_VkImage(underlyingA));

    delete_VkImage(boxed1);
    delete_VkImage(boxed2);
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(underlyingA));
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(underlyingB));
}

TEST(BoxedHandleManagerReverseMapTest, NullUnboxedHandleIsNeverReverseMapped) {
    // Boxing VK_NULL_HANDLE must not register key 0 in the reverse map: it could never be erased
    // (eraseReverseMappingLocked skips 0) and would make lookups of VK_NULL_HANDLE return a
    // stale boxed handle.
    VkImage boxedNull = new_boxed_non_dispatchable_VkImage(VK_NULL_HANDLE);
    ASSERT_NE(VK_NULL_HANDLE, boxedNull);
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(VK_NULL_HANDLE));

    VkImage underlying = reinterpret_cast<VkImage>(0x9999'0000);
    set_boxed_non_dispatchable_VkImage(boxedNull, underlying);
    EXPECT_EQ(boxedNull, unboxed_to_boxed_non_dispatchable_VkImage(underlying));
    set_boxed_non_dispatchable_VkImage(boxedNull, VK_NULL_HANDLE);
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(underlying));
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(VK_NULL_HANDLE));

    delete_VkImage(boxedNull);
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkImage(VK_NULL_HANDLE));
}

TEST(BoxedHandleManagerReverseMapTest, DelayedRemoveClearsReverseMapping) {
    VkDevice device = reinterpret_cast<VkDevice>(0x1111'0000);
    VkSampler underlying = reinterpret_cast<VkSampler>(0x8888'0000);

    VkSampler boxed = new_boxed_non_dispatchable_VkSampler(underlying);
    EXPECT_EQ(boxed, unboxed_to_boxed_non_dispatchable_VkSampler(underlying));

    delayed_delete_VkSampler(boxed, device, nullptr);
    // Still alive until the delayed removes for the device are processed.
    EXPECT_EQ(boxed, unboxed_to_boxed_non_dispatchable_VkSampler(underlying));

    sBoxedHandleManager.processDelayedRemoves(device);
    EXPECT_EQ(nullptr, sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxed)));
    EXPECT_EQ(VK_NULL_HANDLE, unboxed_to_boxed_non_dispatchable_VkSampler(underlying));
}

TEST(VkDecoderGlobalStateProcessCleanupLeakTest, DestroyCommandPoolDeletesBoxedCommandBuffers) {
    VkDecoderGlobalStateTestPeer peer;
    const uint64_t baselineHandles = sBoxedHandleManager.getHandlesCount();

    static int sDestroyedPools = 0;
    sDestroyedPools = 0;
    VulkanDispatch deviceVk = {};
    deviceVk.vkDestroyCommandPool = [](VkDevice, VkCommandPool, const VkAllocationCallbacks*) {
        ++sDestroyedPools;
    };

    VkDevice unboxedDevice = reinterpret_cast<VkDevice>(0xb001'0000ULL);
    VkCommandPool unboxedPool = reinterpret_cast<VkCommandPool>(0xb002'0000ULL);
    VkCommandPool boxedPool = new_boxed_non_dispatchable_VkCommandPool(unboxedPool);
    VkCommandBuffer unboxedCmdBuf = reinterpret_cast<VkCommandBuffer>(0xb003'0000ULL);
    VkCommandBuffer boxedCmdBuf = new_boxed_VkCommandBuffer(unboxedCmdBuf, &deviceVk);
    BoxedHandleInfo* cmdBufBoxedInfo =
        sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxedCmdBuf));
    ASSERT_NE(nullptr, cmdBufBoxedInfo);
    cmdBufBoxedInfo->readStream =
        std::make_unique<VulkanMemReadingStream>(nullptr, gfxstream::host::FeatureSet{});
    std::shared_ptr<OrderMaintenanceInfo> cmdBufOrder = ordmaint_VkCommandBuffer(boxedCmdBuf);
    ASSERT_NE(nullptr, cmdBufOrder);
    EXPECT_EQ(2, cmdBufOrder.use_count());

    CommandPoolInfo poolInfo;
    poolInfo.device = unboxedDevice;
    poolInfo.boxed = boxedPool;
    poolInfo.cmdBuffers.insert(unboxedCmdBuf);

    std::unordered_map<VkCommandBuffer, CommandBufferInfo> cmdBufferInfos;
    CommandBufferInfo& cmdInfo = cmdBufferInfos[unboxedCmdBuf];
    cmdInfo.device = unboxedDevice;
    cmdInfo.cmdPool = unboxedPool;
    cmdInfo.boxed = boxedCmdBuf;

    peer.destroyCommandPoolWithExclusiveInfo(unboxedDevice, &deviceVk, unboxedPool, poolInfo,
                                             cmdBufferInfos);
    delete_VkCommandPool(boxedPool);

    EXPECT_EQ(1, sDestroyedPools);
    EXPECT_TRUE(cmdBufferInfos.empty());
    EXPECT_EQ(nullptr, sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxedCmdBuf)));
    EXPECT_EQ(1, cmdBufOrder.use_count());
    EXPECT_EQ(baselineHandles, sBoxedHandleManager.getHandlesCount());
}

TEST(VkDecoderGlobalStateProcessCleanupLeakTest, DeviceCleanupDeletesAllExtractedBoxedHandles) {
    VkDecoderGlobalStateTestPeer peer;
    const uint64_t baselineHandles = sBoxedHandleManager.getHandlesCount();

    static int sDestroyCalls = 0;
    sDestroyCalls = 0;
    // Direct ICD loading leaves non-global entry points (such as vkDestroyDevice)
    // null on the global dispatch table; cleanup must use the device dispatch.
    VulkanDispatch globalVk = {};
    VulkanDispatch deviceVk = {};
    deviceVk.vkDestroyDevice = [](VkDevice, const VkAllocationCallbacks*) { ++sDestroyCalls; };
    deviceVk.vkDeviceWaitIdle = [](VkDevice) -> VkResult { return VK_SUCCESS; };
    deviceVk.vkDestroyBuffer = [](VkDevice, VkBuffer, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkFreeMemory = [](VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkFreeCommandBuffers = [](VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) {
        ++sDestroyCalls;
    };
    deviceVk.vkDestroyCommandPool = [](VkDevice, VkCommandPool, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkDestroyDescriptorPool =
        [](VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) { ++sDestroyCalls; };
    deviceVk.vkDestroyDescriptorSetLayout =
        [](VkDevice, VkDescriptorSetLayout layout, const VkAllocationCallbacks*) {
            if (layout != VK_NULL_HANDLE) ++sDestroyCalls;
        };
    deviceVk.vkDestroyDescriptorUpdateTemplate =
        [](VkDevice, VkDescriptorUpdateTemplate, const VkAllocationCallbacks*) { ++sDestroyCalls; };
    deviceVk.vkDestroySamplerYcbcrConversion =
        [](VkDevice, VkSamplerYcbcrConversion, const VkAllocationCallbacks*) { ++sDestroyCalls; };
    deviceVk.vkDestroyShaderModule = [](VkDevice, VkShaderModule, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkDestroyPipeline = [](VkDevice, VkPipeline, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkDestroyPipelineCache = [](VkDevice, VkPipelineCache, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkDestroyPipelineLayout =
        [](VkDevice, VkPipelineLayout layout, const VkAllocationCallbacks*) {
            if (layout != VK_NULL_HANDLE) ++sDestroyCalls;
        };
    deviceVk.vkDestroyFramebuffer = [](VkDevice, VkFramebuffer, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };
    deviceVk.vkDestroyRenderPass = [](VkDevice, VkRenderPass, const VkAllocationCallbacks*) {
        ++sDestroyCalls;
    };

    VkDevice unboxedDevice = reinterpret_cast<VkDevice>(0xc001'0000ULL);
    CommandBufferCleanupProbe probe =
        peer.registerDeviceObjectsForCleanupTest(&globalVk, &deviceVk, unboxedDevice);

    InstanceObjects::DeviceObjects deviceObjects =
        peer.extractDeviceObjectsForCleanupTest(unboxedDevice);
    EXPECT_EQ(0u, peer.descriptorUpdateTemplateInfoCount());
    EXPECT_EQ(0u, peer.samplerYcbcrConversionInfoCount());

    peer.destroyDeviceObjects(deviceObjects);

    EXPECT_EQ(14, sDestroyCalls);
    EXPECT_EQ(nullptr, sBoxedHandleManager.get(reinterpret_cast<uint64_t>(probe.boxedCmdBuf)));
    ASSERT_NE(nullptr, probe.cmdBufOrder);
    EXPECT_EQ(1, probe.cmdBufOrder.use_count());
    EXPECT_EQ(baselineHandles, sBoxedHandleManager.getHandlesCount());
}

TEST(VkDecoderGlobalStateProcessCleanupLeakTest,
     EnumeratePhysicalDevicesReturnsInstanceOwnedHandlesUntilInstanceDestroyed) {
    VkDecoderGlobalStateTestPeer peer;
    const uint64_t baselineHandles = sBoxedHandleManager.getHandlesCount();

    static int sDestroyedInstances = 0;
    sDestroyedInstances = 0;
    // Direct ICD loading leaves vkDestroyInstance null on the global dispatch
    // table; instance cleanup must dispatch through the boxed instance table.
    VulkanDispatch globalVk = {};
    VulkanDispatch instanceVk = {};
    instanceVk.vkDestroyInstance = [](VkInstance, const VkAllocationCallbacks*) {
        ++sDestroyedInstances;
    };

    VkInstance unboxedInstance = reinterpret_cast<VkInstance>(0xd001'0000ULL);
    const std::vector<VkPhysicalDevice> unboxedPhysicalDevices = {
        reinterpret_cast<VkPhysicalDevice>(0xd002'0000ULL),
        reinterpret_cast<VkPhysicalDevice>(0xd003'0000ULL),
    };
    const std::vector<VkPhysicalDevice> boxedPhysicalDevices = {
        new_boxed_VkPhysicalDevice(unboxedPhysicalDevices[0], &instanceVk),
        new_boxed_VkPhysicalDevice(unboxedPhysicalDevices[1], &instanceVk),
    };
    VkInstance boxedInstance = peer.registerInstanceWithPhysicalDevicesForCleanupTest(
        &globalVk, &instanceVk, unboxedInstance, unboxedPhysicalDevices, boxedPhysicalDevices);
    EXPECT_EQ(2u, peer.physicalDeviceInfoCount());

    // Count-only query reports every physical device without writing handles.
    uint32_t count = 0;
    EXPECT_EQ(VK_SUCCESS, peer.enumeratePhysicalDevices(boxedInstance, &count, nullptr));
    EXPECT_EQ(2u, count);

    // An array that is too small gets only as many handles as it can hold,
    // the count is updated to the number written and VK_INCOMPLETE returned.
    VkPhysicalDevice out[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    count = 1;
    EXPECT_EQ(VK_INCOMPLETE, peer.enumeratePhysicalDevices(boxedInstance, &count, out));
    EXPECT_EQ(1u, count);
    EXPECT_EQ(boxedPhysicalDevices[0], out[0]);
    EXPECT_EQ(VK_NULL_HANDLE, out[1]);

    // Repeated enumeration on the same instance returns the same boxed
    // handles and does not box new ones.
    for (int i = 0; i < 2; ++i) {
        out[0] = out[1] = VK_NULL_HANDLE;
        count = 2;
        EXPECT_EQ(VK_SUCCESS, peer.enumeratePhysicalDevices(boxedInstance, &count, out));
        EXPECT_EQ(2u, count);
        EXPECT_EQ(boxedPhysicalDevices[0], out[0]);
        EXPECT_EQ(boxedPhysicalDevices[1], out[1]);
    }
    EXPECT_EQ(baselineHandles + 3, sBoxedHandleManager.getHandlesCount());

    // Destroying the instance releases its physical devices with it.
    InstanceObjects instanceObjects = peer.extractInstanceObjectsForCleanupTest(unboxedInstance);
    EXPECT_EQ(0u, peer.physicalDeviceInfoCount());
    peer.destroyInstanceObjects(instanceObjects);

    EXPECT_EQ(1, sDestroyedInstances);
    for (VkPhysicalDevice boxedPhysicalDevice : boxedPhysicalDevices) {
        EXPECT_EQ(nullptr,
                  sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxedPhysicalDevice)));
    }
    EXPECT_EQ(nullptr, sBoxedHandleManager.get(reinterpret_cast<uint64_t>(boxedInstance)));
    EXPECT_EQ(baselineHandles, sBoxedHandleManager.getHandlesCount());
}

class VkDecoderGlobalStateSnapshotImageFlagsTest : public ::testing::Test {
   protected:
    static constexpr VkImageUsageFlags kTransferUsage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    void SetUp() override {
        mBaselineHandles = sBoxedHandleManager.getHandlesCount();
        sLastCreatedUsage = 0;
        sLastCreatedStencilUsage = 0;
        sLastMemReqUsage = 0;
        sLastMemReqStencilUsage = 0;
        sFormatProps1Calls = 0;
        sFormatProps2Calls = 0;
        sFormatProps2KhrCalls = 0;
        sAllowTransferUsage = true;
        sTransferUnsupportedDrmModifier.reset();
        sFormatProps = {
            .maxExtent = {4096, 4096, 256},
            .maxMipLevels = 12,
            .maxArrayLayers = 256,
            .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
            .maxResourceSize = 1ULL << 30,
        };
        sExternalFeatures =
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
        sCompatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        sObservedExternalHandleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(0);
        sObservedFormatListViewCount = 0;
        sObservedDrmModifier = 0;
        sObservedStencilUsage = 0;
        sNextImageId = 0xe100'0000ULL;

        mInstanceVk.vkDestroyInstance = [](VkInstance, const VkAllocationCallbacks*) {};
        mDeviceVk.vkDestroyDevice = [](VkDevice, const VkAllocationCallbacks*) {};

        // Device dispatch intentionally leaves physical-device entry points null, matching
        // vkGetDeviceProcAddr behavior per the Vulkan specification.
        mDeviceVk.vkDeviceWaitIdle = [](VkDevice) -> VkResult { return VK_SUCCESS; };
        mDeviceVk.vkCreateImage = [](VkDevice, const VkImageCreateInfo* pCreateInfo,
                                     const VkAllocationCallbacks*, VkImage* pImage) -> VkResult {
            sLastCreatedUsage = pCreateInfo->usage;
            sLastCreatedStencilUsage = getStencilUsage(pCreateInfo);
            *pImage = reinterpret_cast<VkImage>(sNextImageId++);
            return VK_SUCCESS;
        };
        mDeviceVk.vkDestroyImage = [](VkDevice, VkImage, const VkAllocationCallbacks*) {};
        mDeviceVk.vkDestroyDescriptorSetLayout =
            [](VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {};
        mDeviceVk.vkDestroyPipelineLayout =
            [](VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {};
        mDeviceVk.vkGetDeviceImageMemoryRequirements =
            [](VkDevice, const VkDeviceImageMemoryRequirements* pInfo,
               VkMemoryRequirements2* pMemoryRequirements) {
                sLastMemReqUsage = pInfo->pCreateInfo->usage;
                sLastMemReqStencilUsage = getStencilUsage(pInfo->pCreateInfo);
                pMemoryRequirements->memoryRequirements.size =
                    (pInfo->pCreateInfo->usage & kTransferUsage) == kTransferUsage ? 8192 : 4096;
                pMemoryRequirements->memoryRequirements.alignment = 256;
                pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x1;
            };

        mInstanceVk.vkGetPhysicalDeviceImageFormatProperties =
            [](VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling, VkImageUsageFlags usage,
               VkImageCreateFlags, VkImageFormatProperties* pProps) -> VkResult {
            ++sFormatProps1Calls;
            if (!sAllowTransferUsage && (usage & kTransferUsage) != 0) {
                return VK_ERROR_FORMAT_NOT_SUPPORTED;
            }
            *pProps = sFormatProps;
            return VK_SUCCESS;
        };
        mInstanceVk.vkGetPhysicalDeviceImageFormatProperties2 = &onGetImageFormatProperties2;
        mInstanceVk.vkGetPhysicalDeviceImageFormatProperties2KHR = &onGetImageFormatProperties2Khr;

        mPeer.setSnapshotsEnabled(true);
    }

    void TearDown() override {
        InstanceObjects instanceObjects = mPeer.extractInstanceObjectsForCleanupTest(mUnboxedInst);
        mPeer.destroyInstanceObjects(instanceObjects);
        EXPECT_EQ(mBaselineHandles, sBoxedHandleManager.getHandlesCount());
    }

    VkDevice registerRig(uint32_t apiVersion = VK_API_VERSION_1_1,
                         std::vector<std::string> exts = {}) {
        return mPeer.registerSnapshotImageTestRig(&mGlobalVk, &mInstanceVk, &mDeviceVk,
                                                  mUnboxedInst, mUnboxedPhysDev, mUnboxedDev,
                                                  apiVersion, std::move(exts));
    }

    static VkImageCreateInfo makeBaseImageCreateInfo(VkImageUsageFlags usage) {
        return VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {64, 64, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }

    static VkImageUsageFlags getStencilUsage(const VkImageCreateInfo* pCreateInfo) {
        const auto* stencilUsageCi = vk_find_struct<VkImageStencilUsageCreateInfo>(pCreateInfo);
        return stencilUsageCi ? stencilUsageCi->stencilUsage : 0;
    }

    static VkResult handleImageFormatProperties2(const VkPhysicalDeviceImageFormatInfo2* pInfo,
                                                 VkImageFormatProperties2* pProps) {
        std::optional<uint64_t> drmModifier;
        for (const auto* node = reinterpret_cast<const VkBaseInStructure*>(pInfo->pNext); node;
             node = node->pNext) {
            // VkExternalMemoryImageCreateInfo is only valid on VkImageCreateInfo, never on
            // VkPhysicalDeviceImageFormatInfo2.
            EXPECT_NE(VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, node->sType);
            if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO) {
                sObservedExternalHandleType =
                    reinterpret_cast<const VkPhysicalDeviceExternalImageFormatInfo*>(node)
                        ->handleType;
            } else if (node->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
                sObservedFormatListViewCount =
                    reinterpret_cast<const VkImageFormatListCreateInfo*>(node)->viewFormatCount;
            } else if (node->sType ==
                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT) {
                drmModifier =
                    reinterpret_cast<const VkPhysicalDeviceImageDrmFormatModifierInfoEXT*>(node)
                        ->drmFormatModifier;
                sObservedDrmModifier = *drmModifier;
            } else if (node->sType == VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO) {
                sObservedStencilUsage =
                    reinterpret_cast<const VkImageStencilUsageCreateInfo*>(node)->stencilUsage;
            }
        }
        const bool rejectTransferUsage =
            !sAllowTransferUsage ||
            (drmModifier.has_value() && drmModifier == sTransferUnsupportedDrmModifier);
        if (rejectTransferUsage && (pInfo->usage & kTransferUsage) != 0) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        pProps->imageFormatProperties = sFormatProps;
        if (auto* extProps = vk_find_struct<VkExternalImageFormatProperties>(pProps)) {
            extProps->externalMemoryProperties.externalMemoryFeatures = sExternalFeatures;
            extProps->externalMemoryProperties.compatibleHandleTypes = sCompatibleHandleTypes;
        }
        return VK_SUCCESS;
    }

    static VkResult onGetImageFormatProperties2(VkPhysicalDevice,
                                                const VkPhysicalDeviceImageFormatInfo2* pInfo,
                                                VkImageFormatProperties2* pProps) {
        ++sFormatProps2Calls;
        return handleImageFormatProperties2(pInfo, pProps);
    }

    static VkResult onGetImageFormatProperties2Khr(VkPhysicalDevice,
                                                   const VkPhysicalDeviceImageFormatInfo2* pInfo,
                                                   VkImageFormatProperties2* pProps) {
        ++sFormatProps2KhrCalls;
        return handleImageFormatProperties2(pInfo, pProps);
    }

    VkDecoderGlobalStateTestPeer mPeer;
    VulkanDispatch mGlobalVk = {};
    VulkanDispatch mInstanceVk = {};
    VulkanDispatch mDeviceVk = {};
    gfxstream::base::BumpPool mPool;
    uint64_t mBaselineHandles = 0;

    VkInstance mUnboxedInst = reinterpret_cast<VkInstance>(0xe001'0000ULL);
    VkPhysicalDevice mUnboxedPhysDev = reinterpret_cast<VkPhysicalDevice>(0xe002'0000ULL);
    VkDevice mUnboxedDev = reinterpret_cast<VkDevice>(0xe003'0000ULL);

    static inline VkImageUsageFlags sLastCreatedUsage = 0;
    static inline VkImageUsageFlags sLastCreatedStencilUsage = 0;
    static inline VkImageUsageFlags sLastMemReqUsage = 0;
    static inline VkImageUsageFlags sLastMemReqStencilUsage = 0;
    static inline int sFormatProps1Calls = 0;
    static inline int sFormatProps2Calls = 0;
    static inline int sFormatProps2KhrCalls = 0;
    static inline bool sAllowTransferUsage = true;
    static inline std::optional<uint64_t> sTransferUnsupportedDrmModifier;
    static inline VkImageFormatProperties sFormatProps = {};
    static inline VkExternalMemoryFeatureFlags sExternalFeatures = 0;
    static inline VkExternalMemoryHandleTypeFlags sCompatibleHandleTypes = 0;
    static inline VkExternalMemoryHandleTypeFlagBits sObservedExternalHandleType =
        static_cast<VkExternalMemoryHandleTypeFlagBits>(0);
    static inline uint32_t sObservedFormatListViewCount = 0;
    static inline uint64_t sObservedDrmModifier = 0;
    static inline VkImageUsageFlags sObservedStencilUsage = 0;
    static inline uint64_t sNextImageId = 0xe100'0000ULL;
};

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       UsesInstanceDispatchAndAddsTransferFlagsWhenSupported) {
    VkDevice boxedDevice = registerRig();
    ASSERT_EQ(nullptr, mDeviceVk.vkGetPhysicalDeviceImageFormatProperties);

    VkImageCreateInfo ci = makeBaseImageCreateInfo(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                   VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImage boxedImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage));
    EXPECT_EQ(1, sFormatProps1Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage,
              sLastCreatedUsage);

    // If transfer usage only supports 1x samples, 4x MSAA images must keep their original usage.
    VkImageCreateInfo msaaCi = ci;
    msaaCi.samples = VK_SAMPLE_COUNT_4_BIT;
    VkImage boxedMsaaImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &msaaCi, &boxedMsaaImage));
    EXPECT_EQ(2, sFormatProps1Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, sLastCreatedUsage);

    sAllowTransferUsage = false;
    VkImage boxedImageUnsupported = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImageUnsupported));
    EXPECT_EQ(3, sFormatProps1Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, sLastCreatedUsage);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       SkipsRedundantFormatQueryWhenTransferFlagsAlreadySet) {
    VkDevice boxedDevice = registerRig();
    VkImageCreateInfo ci =
        makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage);

    VkImage boxedImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(0, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastCreatedUsage);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       QueriesExternalImageFormatProperties2WhenExternalMemoryPresent) {
    VkDevice boxedDevice = registerRig(VK_API_VERSION_1_1);

    VkExternalMemoryImageCreateInfo extMemCi = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkImageCreateInfo ci = makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT);
    ci.pNext = &extMemCi;

    // Driver reports format unsupported for transfer usage with this external handle type.
    sAllowTransferUsage = false;
    VkImage boxedImage1 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage1));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(1, sFormatProps2Calls);
    EXPECT_EQ(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, sObservedExternalHandleType);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT, sLastCreatedUsage);

    // Driver returns VK_SUCCESS, but externalMemoryFeatures == 0 (handle type not usable with
    // transfer usage).
    sAllowTransferUsage = true;
    sExternalFeatures = 0;
    VkImage boxedImage2 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage2));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(2, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT, sLastCreatedUsage);

    // Driver reports external handle type is importable/exportable and compatible with transfer
    // usage.
    sExternalFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
    sCompatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkImage boxedImage3 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage3));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(3, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastCreatedUsage);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       UsesImageFormatProperties2KHRAndForwardsCompatiblePNextStructs) {
    VkDevice boxedDevice =
        registerRig(VK_API_VERSION_1_0, {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME});

    const VkFormat viewFormats[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB};
    VkExternalMemoryImageCreateInfo extMemCi = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkImageFormatListCreateInfo formatListCi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
        .pNext = &extMemCi,
        .viewFormatCount = 2,
        .pViewFormats = viewFormats,
    };
    VkImageCreateInfo ci = makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT);
    ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    ci.pNext = &formatListCi;

    VkImage boxedImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(0, sFormatProps2Calls);
    EXPECT_EQ(1, sFormatProps2KhrCalls);
    EXPECT_EQ(2u, sObservedFormatListViewCount);
    EXPECT_EQ(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, sObservedExternalHandleType);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastCreatedUsage);
    // Original guest pNext chain must remain intact.
    EXPECT_EQ(&extMemCi, formatListCi.pNext);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       MirrorsTransferUsageAdjustmentInGetDeviceImageMemoryRequirements) {
    VkDevice boxedDevice = registerRig();

    VkImageCreateInfo ci = makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT);
    VkDeviceImageMemoryRequirements reqInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
        .pNext = nullptr,
        .pCreateInfo = &ci,
        .planeAspect = VK_IMAGE_ASPECT_NONE,
    };
    VkMemoryRequirements2 memReqs = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
    };

    mPeer.getDeviceImageMemoryRequirements(&mPool, boxedDevice, &reqInfo, &memReqs);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastMemReqUsage);
    EXPECT_EQ(8192u, memReqs.memoryRequirements.size);

    // When driver rejects transfer usage, memory requirements query uses original usage too.
    sAllowTransferUsage = false;
    mPeer.getDeviceImageMemoryRequirements(&mPool, boxedDevice, &reqInfo, &memReqs);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT, sLastMemReqUsage);
    EXPECT_EQ(4096u, memReqs.memoryRequirements.size);

    // When snapshots are disabled, no transfer flags are added.
    sAllowTransferUsage = true;
    mPeer.setSnapshotsEnabled(false);
    mPeer.getDeviceImageMemoryRequirements(&mPool, boxedDevice, &reqInfo, &memReqs);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT, sLastMemReqUsage);
    EXPECT_EQ(4096u, memReqs.memoryRequirements.size);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       HandlesDrmFormatModifiersAndSkipsUnrecognizedPNextStructs) {
    VkDevice boxedDevice = registerRig();

    constexpr uint64_t kModifier = 0x0100000000000001ULL;
    VkImageDrmFormatModifierListCreateInfoEXT drmListCi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .pNext = nullptr,
        .drmFormatModifierCount = 1,
        .pDrmFormatModifiers = &kModifier,
    };
    VkImageCreateInfo drmCi = makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT);
    drmCi.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    drmCi.pNext = &drmListCi;

    VkImage boxedDrmImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &drmCi, &boxedDrmImage));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(1, sFormatProps2Calls);
    EXPECT_EQ(kModifier, sObservedDrmModifier);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastCreatedUsage);

    // Unrecognized VkImageCreateInfo-only structs (e.g. VkImageSwapchainCreateInfoKHR) fall back
    // conservatively without querying or mutating usage.
    VkImageSwapchainCreateInfoKHR swapchainCi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .swapchain = VK_NULL_HANDLE,
    };
    VkImageCreateInfo swapImgCi = makeBaseImageCreateInfo(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    swapImgCi.pNext = &swapchainCi;

    VkImage boxedSwapImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &swapImgCi, &boxedSwapImage));
    EXPECT_EQ(0, sFormatProps1Calls);
    EXPECT_EQ(1, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, sLastCreatedUsage);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest, AddsTransferFlagsToSeparateStencilUsage) {
    VkDevice boxedDevice = registerRig();

    constexpr VkImageUsageFlags kDepthStencilUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageStencilUsageCreateInfo stencilUsageCi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .stencilUsage = kDepthStencilUsage,
    };
    VkImageCreateInfo ci = makeBaseImageCreateInfo(kDepthStencilUsage);
    ci.format = VK_FORMAT_D24_UNORM_S8_UINT;
    ci.pNext = &stencilUsageCi;

    VkImage boxedImage1 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage1));
    EXPECT_EQ(1, sFormatProps2Calls);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sObservedStencilUsage);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastCreatedUsage);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastCreatedStencilUsage);
    // Original guest pNext chain must remain intact.
    EXPECT_EQ(kDepthStencilUsage, stencilUsageCi.stencilUsage);

    VkDeviceImageMemoryRequirements reqInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
        .pNext = nullptr,
        .pCreateInfo = &ci,
        .planeAspect = VK_IMAGE_ASPECT_NONE,
    };
    VkMemoryRequirements2 memReqs = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
    };
    mPeer.getDeviceImageMemoryRequirements(&mPool, boxedDevice, &reqInfo, &memReqs);
    EXPECT_EQ(2, sFormatProps2Calls);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastMemReqUsage);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastMemReqStencilUsage);

    // The stencil usage still needs the transfer flags when only the depth usage has them.
    ci.usage = kDepthStencilUsage | kTransferUsage;
    VkImage boxedImage2 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage2));
    EXPECT_EQ(3, sFormatProps2Calls);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastCreatedUsage);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastCreatedStencilUsage);

    // No driver query is needed once all aspects already request the transfer flags.
    stencilUsageCi.stencilUsage = kDepthStencilUsage | kTransferUsage;
    VkImage boxedImage3 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage3));
    EXPECT_EQ(3, sFormatProps2Calls);
    EXPECT_EQ(kDepthStencilUsage | kTransferUsage, sLastCreatedStencilUsage);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest, SkipsSparseImagesRequestedWithCreateFlags2) {
    VkDevice boxedDevice = registerRig();

    VkImageCreateFlags2CreateInfoKHR flags2Ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_FLAGS_2_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = VK_IMAGE_CREATE_2_SPARSE_BINDING_BIT_KHR,
    };
    VkImageCreateInfo ci = makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT);
    ci.pNext = &flags2Ci;

    VkImage boxedSparseImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedSparseImage));
    EXPECT_EQ(0, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT, sLastCreatedUsage);

    // Non-sparse 64-bit create flags do not prevent adding the transfer flags.
    flags2Ci.flags = VK_IMAGE_CREATE_2_MUTABLE_FORMAT_BIT_KHR;
    VkImage boxedImage = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage));
    EXPECT_EQ(1, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastCreatedUsage);
}

TEST_F(VkDecoderGlobalStateSnapshotImageFlagsTest,
       RequiresAllListedDrmFormatModifiersToSupportTransferFlags) {
    VkDevice boxedDevice = registerRig();

    const uint64_t modifiers[2] = {0x0100000000000001ULL, 0x0100000000000002ULL};
    VkImageDrmFormatModifierListCreateInfoEXT drmListCi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .pNext = nullptr,
        .drmFormatModifierCount = 2,
        .pDrmFormatModifiers = modifiers,
    };
    VkImageCreateInfo ci = makeBaseImageCreateInfo(VK_IMAGE_USAGE_SAMPLED_BIT);
    ci.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    ci.pNext = &drmListCi;

    // The driver may select any listed modifier, so a single modifier without transfer support
    // keeps the original usage.
    sTransferUnsupportedDrmModifier = modifiers[1];
    VkImage boxedImage1 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage1));
    EXPECT_EQ(2, sFormatProps2Calls);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT, sLastCreatedUsage);

    sTransferUnsupportedDrmModifier.reset();
    VkImage boxedImage2 = VK_NULL_HANDLE;
    ASSERT_EQ(VK_SUCCESS, mPeer.createImage(&mPool, boxedDevice, &ci, &boxedImage2));
    EXPECT_EQ(4, sFormatProps2Calls);
    EXPECT_EQ(modifiers[1], sObservedDrmModifier);
    EXPECT_EQ(VK_IMAGE_USAGE_SAMPLED_BIT | kTransferUsage, sLastCreatedUsage);
}

}  // namespace
}  // namespace vk
}  // namespace host
}  // namespace gfxstream

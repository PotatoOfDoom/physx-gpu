#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define VK(x) do { VkResult r_=(x); if(r_!=VK_SUCCESS){std::fprintf(stderr,"%s -> %d line %d\n",#x,int(r_),__LINE__);std::exit(2);} } while(0)
static std::vector<char> rd(const char* p){FILE*f=std::fopen(p,"rb");if(!f)std::exit(2);std::fseek(f,0,SEEK_END);long n=std::ftell(f);std::fseek(f,0,SEEK_SET);std::vector<char> b((size_t(n)));if(std::fread(b.data(),1,size_t(n),f)!=size_t(n))std::exit(2);std::fclose(f);return b;}
int main(){
    VkInstance inst; VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo=&app; VK(vkCreateInstance(&ii,nullptr,&inst));
    uint32_t n=0; VK(vkEnumeratePhysicalDevices(inst,&n,nullptr)); std::vector<VkPhysicalDevice> d(n); VK(vkEnumeratePhysicalDevices(inst,&n,d.data()));
    VkPhysicalDevice phys=d[0]; uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qs.data());
    uint32_t qf=0; for(uint32_t i=0;i<qn;++i) if(qs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;}
    float pr=1.f; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex=qf; qi.queueCount=1; qi.pQueuePriorities=&pr;
    VkPhysicalDeviceVulkan12Features f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f.bufferDeviceAddress=VK_TRUE;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext=&f; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi;
    VkDevice dev; VK(vkCreateDevice(phys,&di,nullptr,&dev)); VkQueue q; vkGetDeviceQueue(dev,qf,0,&q);
    auto mk=[&](VkDeviceSize sz,VkDeviceAddress&addr,void*&map){ VkBuffer b; VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=sz;
        bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; VK(vkCreateBuffer(dev,&bi,nullptr,&b));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,b,&mr); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
        uint32_t mt=0; for(uint32_t i=0;i<mp.memoryTypeCount;++i) if((mr.memoryTypeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){mt=i;break;}
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fi.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext=&fi; ai.allocationSize=mr.size; ai.memoryTypeIndex=mt;
        VkDeviceMemory m; VK(vkAllocateMemory(dev,&ai,nullptr,&m)); VK(vkBindBufferMemory(dev,b,m,0)); VK(vkMapMemory(dev,m,0,VK_WHOLE_SIZE,0,&map));
        VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; dai.buffer=b; addr=vkGetBufferDeviceAddress(dev,&dai); };
    VkDeviceAddress sa,da; void *sm,*dm; mk(256,sa,sm); mk(256,da,dm); std::memset(dm,0,256);
    auto spv=rd("ultest.spv"); VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=spv.size(); smi.pCode=(const uint32_t*)spv.data();
    VkShaderModule sm2; VK(vkCreateShaderModule(dev,&smi,nullptr,&sm2));
    struct P{uint64_t src,dst;} push{ (uint64_t)sa, (uint64_t)da };
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(P)}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; VkPipelineLayout lay; VK(vkCreatePipelineLayout(dev,&pli,nullptr,&lay));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module=sm2; cpi.stage.pName="ulMain"; cpi.layout=lay;
    VkPipeline pipe; VK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpi,nullptr,&pipe));
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cp.queueFamilyIndex=qf; VkCommandPool pool; VK(vkCreateCommandPool(dev,&cp,nullptr,&pool));
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbi.commandPool=pool; cbi.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount=1;
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev,&cbi,&cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cb,&bi)); vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdPushConstants(cb,lay,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(P),&push); vkCmdDispatch(cb,1,1,1); VK(vkEndCommandBuffer(cb));
    VkFence fe; VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK(vkCreateFence(dev,&fci,nullptr,&fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    VK(vkQueueSubmit(q,1,&si,fe)); VK(vkWaitForFences(dev,1,&fe,VK_TRUE,~0ull));
    const uint64_t* r=(const uint64_t*)dm;
    const char* nm[5]={"addr (unaligned)","& ~3UL","& ~3ull","& ~uint64_t(3)","- (addr % 4UL)"};
    for(int i=0;i<5;++i) std::printf("  %-18s = 0x%016llx%s\n", nm[i], (unsigned long long)r[i],
        (i>0 && (r[i]>>32)!=(r[0]>>32)) ? "   <-- TOP HALF LOST" : "");
    return 0;
}

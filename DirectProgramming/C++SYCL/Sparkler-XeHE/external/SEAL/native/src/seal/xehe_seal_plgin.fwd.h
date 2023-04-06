#ifndef _XEHE_SEAL_PLGIN_FWD_H_
#define _XEHE_SEAL_PLGIN_FWD_H_

#ifdef SEAL_USE_INTEL_XEHE
#include <XeHE.hpp>

namespace xehe{
    namespace plgin{

        class XeHECiphertext;

        template<typename T>
        class XEHEvaluator;

        class XeHECiphertext
        {
            public:
            XeHECiphertext()
            {
            }


            XeHECiphertext(const XeHECiphertext &copy)
            {
                *this = copy;
            }

            XeHECiphertext &operator=(const XeHECiphertext &assign)
            {
        // if assigning from non-gpu to already gpu
                if (is_on_gpu() &&  !assign.is_on_gpu())
                {
                    gpu_mem64_ = nullptr;
                    gpu_mem32_ = nullptr;
                }            

                on_gpu(assign.is_on_gpu());
                set_prime64(assign.is_prime64());            

                // allocate and copy gpu buffer
                if (assign.is_on_gpu_mem())
                {
                    //std::cout << "assign " << std::endl;
                    // self will be on gpu since assign is on gpu
                    // allocation conditioned on availablity or/and size
                    auto new_size = assign.get_size();
                    alloc_gpu_memory(new_size);
                    deep_copy(assign);  
                }

                return(*this);
            }

            void resize(size_t new_size, bool discard_alloc = false, bool wait = false)
            {
                if ( new_size > 0 && is_on_gpu_mem())
                {
                if (is_prime64())
                {
                    auto old_gpu_mem = get_gpu_mem<uint64_t>();
                    //std::cout << "resize " << new_size << " " << discard_alloc << std::endl;
                    alloc_gpu_memory(new_size);
                    if (!discard_alloc && old_gpu_mem!=nullptr)
                    {
                        deep_copy<uint64_t>(old_gpu_mem, wait);    
                    }         
                }
                else
                {
                    auto old_gpu_mem = get_gpu_mem<uint32_t>();
                    alloc_gpu_memory(new_size);

                    if (!discard_alloc && old_gpu_mem!=nullptr)
                    {
                        deep_copy<uint32_t>(old_gpu_mem, wait);    
                    }         
                }   
                }                    
            }

            const void* get_gpu_mem_ptr(void) const
            {
                void * ret = 0;
                if (is_prime64())
                {
                    ret =
                        (get_gpu_mem<uint64_t>() != nullptr && get_gpu_mem<uint64_t>()->is_allocated()) ? get_gpu_mem<uint64_t>().get() : nullptr; 
                }
                else
                {
                    ret = 
                        (get_gpu_mem<uint32_t>() != nullptr && get_gpu_mem<uint32_t>()->is_allocated())
                              ? get_gpu_mem<uint32_t>().get()
                              : nullptr; 
                }
                return((const void*)ret);
            } 

            template<typename T> 
            const std::shared_ptr<xehe::ext::Buffer<T>> &get_gpu_mem(void) const
            {
                if constexpr (sizeof(T) ==8)
                {
                    return(gpu_mem64_);
                }
                else
                {
                    return(gpu_mem32_);
                }
            }

            template<typename T> 
            std::shared_ptr<xehe::ext::Buffer<T>> &get_gpu_mem(void)
            {
                if constexpr (sizeof(T) ==8)
                {
                    return(gpu_mem64_);
                }
                else
                {
                    return(gpu_mem32_);
                }
            }


            size_t get_size(void)
            {
                return((is_prime64()) ? gpu_mem64_->get_size() : gpu_mem32_->get_size());
            }

            inline void set_prime64(bool val = true)
            {
                prime64_ = val;
            }

            inline bool is_prime64(void) const
            {
                return(prime64_);
            }

            inline void on_gpu(bool val = true)
            {
                on_gpu_ = val;
            }

            inline bool is_on_gpu(void) const
            {
                return(on_gpu_);
            }

            inline bool is_on_gpu_mem(void) const
            {
                return(is_on_gpu() && get_gpu_mem_ptr() != nullptr && !is_dirty());
            }

            inline void set_dirty(bool val = true)
            {
                dirty_ = val;
            }

            inline bool is_dirty(void) const
            {
                return(dirty_);
            }

            const void *alloc_gpu_memory(size_t new_size)
            {
                if (new_size > 0 && is_on_gpu())
                {
                    auto gpu_mem = get_gpu_mem_ptr();

                    if (gpu_mem == nullptr || (gpu_mem != nullptr && new_size != get_size()))
                    {
                        if (is_prime64())
                        {
                            gpu_mem64_ = xehe::ext::XeHE_malloc<uint64_t>(new_size);
                        }
                        else
                        {
                            gpu_mem32_ = xehe::ext::XeHE_malloc<uint32_t>(new_size);
                        }
                    }
                }
                                        
                return (get_gpu_mem_ptr());
            }

            inline size_t get_size(void) const
            {
                size_t ret = 0;
                if (get_gpu_mem_ptr() != nullptr)
                {
                    if (is_prime64())
                    {
                        ret = get_gpu_mem<uint64_t>()->get_size();
                    }
                    else
                    {
                        ret = get_gpu_mem<uint32_t>()->get_size();                        
                    }                    
                }
                return(ret);
            }
            
            template<typename T>
            void deep_copy(const std::shared_ptr<xehe::ext::Buffer<T>> &src, bool wait = false)
            {
                if (get_gpu_mem<T>() != nullptr && src != nullptr)
                {
                    //std:: cout << "dc "  << std::endl;                    
                    get_gpu_mem<T>()->deep_copy(src, wait);
                }
            }

            void deep_copy(const XeHECiphertext &src, bool wait = false)
            {
                if (is_prime64())
                {
                    deep_copy<uint64_t>(src.get_gpu_mem<uint64_t>(), wait);
                }
                else
                {
                    deep_copy<uint32_t>(src.get_gpu_mem<uint32_t>(), wait);
                }
            }

            void download(void * host_ptr, size_t download_sz, bool wait = true)
            {
                //std::cout << " download0 " << download_sz << std::endl;
                if (host_ptr != nullptr && download_sz > 0 && is_on_gpu_mem())
                {
                    if (is_prime64())
                    {
                        //std::cout << " download1 " << download_sz << std::endl;
                        get_gpu_mem<uint64_t>()->get_data_adapter<uint64_t>((uint64_t *)host_ptr, download_sz, wait);
                    }
                    else
                    {
                        get_gpu_mem<uint32_t>()->get_data_adapter<uint32_t>((uint32_t *)host_ptr, download_sz, wait);
                    }
                }
            }

            void upload(const void *host_ptr, size_t upload_sz, bool wait = false)
            {
                //std:: cout << "upload0 " << upload_sz  << std::endl;
                if (host_ptr !=nullptr && upload_sz > 0 && is_on_gpu() && !is_on_gpu_mem())
                {
                    //std:: cout << "upload alloc " << upload_sz  << std::endl;
                    alloc_gpu_memory(upload_sz);

                    if (is_prime64())
                    {
                        //std:: cout << "set data "  << std::endl;
                        get_gpu_mem<uint64_t>()->set_data_adapter<uint64_t>((const uint64_t *)host_ptr, upload_sz, wait);
                    }
                    else
                    {
                        get_gpu_mem<uint32_t>()->set_data_adapter<uint32_t>((const uint32_t *)host_ptr, upload_sz, wait);
                    }

                    set_dirty(false);
                }
            }


            protected:
            std::shared_ptr<xehe::ext::Buffer<uint64_t>> gpu_mem64_ = nullptr;
            std::shared_ptr<xehe::ext::Buffer<uint32_t>> gpu_mem32_ = nullptr;
            bool dirty_ = false;
            bool prime64_ = true;
            
#ifdef GPU_DEFAULT
            bool on_gpu_ = true;
#else
            bool on_gpu_ = false;
#endif            
        };


    }
}

#endif //#ifdef SEAL_USE_INTEL_XEHE

#endif //#ifndef _XEHE_SEAL_PLGIN_FWD_H_
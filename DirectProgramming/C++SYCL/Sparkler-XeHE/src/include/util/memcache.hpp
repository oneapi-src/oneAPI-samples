#ifndef XeHE_MEMCACHE_H
#define XeHE_MEMCACHE_H

#include <mutex>

namespace xehe{

    typedef struct {
        void* dev_ptr_;
        size_t capacity_;
    } FREE_STRUCT;

    typedef struct {
        size_t capacity_;
        union {
            uint64_t retained_ : 1;
        };
    } ALLOC_STRUCT;

    static bool sort_free_list(const FREE_STRUCT &x, const FREE_STRUCT &y)
    {
        return(x.capacity_ < y.capacity_);
    }

    class Memcache{
    public:
        Memcache(const cl::sycl::queue & q):q_(q){};

        ~Memcache(void){
            dealloc();
        }

        void activate_cache(void)
        {
            pool_memcache_active() = true;
        }

        void deactivate_cache(void)
        {   
            pool_memcache_active() = false;
        }

        void get_memory_cache_stat(size_t &alloced, size_t &freed)
        {
            //  need to do that once per Evaluator session
            // otherwise it takes a lot of time
            //std::sort (get_free_pool().begin(), get_free_pool().end(), XeHE::sort_free_list);


            alloced = get_alloc_pool().size();
            freed = get_free_pool().size();
            size_t data_type_sz = sizeof(uint64_t);
            double alloced_sz = 0, freed_sz = 0;
            for (auto s : get_free_pool())
            {
                freed_sz += s.capacity_;
            }  

            for (auto s : get_alloc_pool())
            {
                alloced_sz += s.second.capacity_;
            }  

            auto alloced_sz_mb = uint64_t((alloced_sz*data_type_sz)/1000000 +0.5);
            auto freed_sz_mb = uint64_t((freed_sz*data_type_sz)/1000000 +0.5);
            std::cout << "Mem cache in MB.\n" << "Free: " << freed_sz_mb
            << " Allocated: " << alloced_sz_mb
            << " Total: " << (freed_sz_mb + alloced_sz_mb)
            << std::endl;
        }


        void pool_free_free(void)
        {
            // acquire free_pool lock to clear
            std::lock_guard<std::mutex> lk(free_lock);
            for (auto s : get_free_pool())
            {
                cl::sycl::free(s.dev_ptr_, get_queue());
            }            
            get_free_pool().clear();
            // free_pool lock will be released out of scope
        }

        void dealloc(void)
        {
            deactivate_cache();
            size_t alloced, freed;
//                get_queue().wait();
            pool_free_free();
            get_memory_cache_stat(alloced, freed);                
        }

        void pool_free(void * data, size_t capacity, bool uncached = false)
        {
            if (!pool_alloc_is_retained(data))
            {
                // acquire free&alloc locks
                std::scoped_lock lock(free_lock, alloc_lock);
                pool_alloc_remove(data);
                pool_free_add(data, capacity, uncached);
                // locks will be released out of scope
                //std::cout << "free add " << std::hex << (void*)data << " " << capacity() << std::endl;
            }
        }
        
        template<typename T>
        T * pool_alloc(size_t buffer_size, size_t & capacity, bool uncached = false)
        {
            T *ret = nullptr;
            if (buffer_size > 0)
            {
                capacity = buffer_size;
                size_t new_capacity = capacity;

                //auto n_fitbuffers = pool_free_num_fitbuffers(buffer_size);
                //std::cout << "n fits "<< n_fitbuffers << " for " << buffer_size << std::endl;

                if (uncached || !pool_memcache_active()){
                    ret = cl::sycl::malloc_device<T>(buffer_size, get_queue());
                }
                else{
                    bool new_malloc = true;
                    {
                        // acquire free&alloc locks
                        std::scoped_lock lock(free_lock, alloc_lock);

                        auto pooled_free_ptr = pool_free_remove(buffer_size, new_capacity);
                        capacity = new_capacity;
                        if (pooled_free_ptr != nullptr){
                            ret = (T*)pooled_free_ptr;
                            pool_alloc_add((void*)ret, capacity);
                            new_malloc = false;
                        }
                        // locks will be released out of scope
                    }
                    if (new_malloc){
                        // std::cout << "Allocating " << buffer_size << " on queue " << std::hex << &q_ << std::dec <<std::endl;
                        ret = cl::sycl::malloc_device<T>(buffer_size, get_queue());
                        // acquire only alloc lock
                        std::lock_guard<std::mutex> lk(alloc_lock);
                        pool_alloc_add((void*)ret, capacity);
                        // lock will be released out of scope
                    }
                }
            }
            else
            {
                std::cout << "Warning: tried to alloc 0 size buffer" << std::endl;
                //throw; 
            }
            return(ret);                
        }

        void pool_alloc_add(void* new_ptr, size_t capacity, uint32_t retained = 0)
        {
            get_alloc_pool()[new_ptr].retained_ = retained;
            get_alloc_pool()[new_ptr].capacity_ = capacity;                                
        }

        void pool_alloc_remove(void* ptr)
        {
            get_alloc_pool().erase(ptr);
        }

        void pool_free_add(void* ptr, size_t capacity, bool uncached = false)
        {

            FREE_STRUCT new_entry;
            new_entry.dev_ptr_ = ptr;
            new_entry.capacity_ = capacity;

            if (ptr != nullptr)
            {
                if (!pool_memcache_active() || uncached)
                {
                    cl::sycl::free(ptr, get_queue());
                //std::cout << "freed " << std::hex << ptr << " " << capacity << std::endl;                                     
                }
                else
                {
                    get_free_pool().push_back(new_entry);
            //std::cout << "freed cached " << std::hex << ptr << " " << capacity << std::endl;
                    if(get_alloc_pool().size() == 0)
                    {
                        //std::cout <<"alloc = 0" << std::endl; 
                        std::sort (get_free_pool().begin(), get_free_pool().end(), sort_free_list);
                    }
                }
            }
            else
            {
                std::cout << "Warning: tried to free nullptr" << std::endl; 
                //throw;
            }

        }

        void* pool_free_remove(size_t size, size_t& capacity)
        {
            void* ret = nullptr;
            for (std::vector<FREE_STRUCT>::iterator it = get_free_pool().begin(); it != get_free_pool().end(); ++it)
            {
                if ((*it).capacity_ >= size)
                {
                    ret = (*it).dev_ptr_;
                    capacity = (*it).capacity_;
                    get_free_pool().erase(it);
                    break;
                }
            }
            return (ret);
        }

        uint32_t pool_alloc_is_retained(void* ptr)
        {
            return((get_alloc_pool().find(ptr) != get_alloc_pool().end()) ? get_alloc_pool()[ptr].retained_ : 0);
        }

        size_t pool_free_num_fitbuffers(size_t size)
        {
            size_t count = 0;
            for (auto s : get_free_pool())
            {
                if (s.capacity_ >= size)
                {
                    ++count;
                }
            }
            return count;
        }

        const bool & pool_memcache_active(void) const
        {
            return(mem_cache_active_);
        }

        bool & pool_memcache_active(void)
        {
            return(mem_cache_active_);
        }

        cl::sycl::queue & get_queue(){
            return (cl::sycl::queue&) q_;
        }

    private:
        
        std::vector<FREE_STRUCT>& get_free_pool(void)
        {
            return(free_pool_);
        }

        std::map<void*, ALLOC_STRUCT>& get_alloc_pool(void)
        {
            return(alloc_pool_);
        }

        // define the state of the cache
        // if at leat 1 buffer is in the alloc cahce it's in flight
        // it switched to not in_flight on delloc call.
        // after that any freeing buffer is  phyiscally freed
        // flashing the cache
        bool mem_cache_active_ = true;
        std::vector<FREE_STRUCT> free_pool_;
        std::map<void*, ALLOC_STRUCT> alloc_pool_;
        const cl::sycl::queue& q_;
        std::mutex free_lock;
        std::mutex alloc_lock;

    }; // class memcache
} // namespace xehe
#endif // #ifdef MEMCACHE
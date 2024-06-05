/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This application demonstrates an approach to the image segmentation
 * trees construction. It is based on Boruvka's MST algorithm.
 * Here's the complete list of references:
 * 1) V. Vineet et al, "Fast Minimum Spanning Tree for
 *    Large Graphs on the GPU";
 * 2) P. Felzenszwalb et al, "Efficient Graph-Based Image Segmentation";
 * 3) A. Ion et al, "Considerations Regarding the Minimum Spanning
 *    Tree Pyramid Segmentation Method".
 */

// System includes.
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// STL includes.
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <list>
#include <deque>
#include <algorithm>
#include <dpct/dpl_utils.hpp>

// Thrust library includes.

// Sample framework includes.
#include <helper_functions.h>
#include <helper_cuda.h>

// Project includes.
#include <common.dp.hpp>

// Kernels.
#include <kernels.dp.hpp>
#include <chrono>

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::list;
using std::deque;

// Very simple von Neumann middle-square prng.  rand() is different across
// various OS platforms, which makes testing and the output inconsistent.
int myrand(void)
{
    static int seed = 72191;
    char sq[22];

    seed *= seed;
    sprintf(sq, "%010d", seed);
    // pull the middle 5 digits out of sq
    sq[8] = 0;
    seed = atoi(&sq[3]);

    return seed;
}

// Simple memory pool class. It is nothing more than array of fixed-sized
// arrays.
template <typename T>
class DeviceMemoryPool
{
    public:
        // The parameters of the constructor are as follows:
        // 1) uint chunkSize --- size of the particular array;
        // 2) uint chunksCount --- number of fixed-sized arrays.
        DeviceMemoryPool(uint chunkSize, uint chunksCount) :
            chunkSize_(chunkSize)
        {
            chunkRawSize_ = (chunkSize * sizeof(T) + 511) & ~511;

            try
            {
                basePtr_ = dpct::malloc_device(chunkRawSize_ * chunksCount);
            }
            catch (std::system_error &e)
            {
                cout << "Pool memory allocation failed (" << e.what() << ")"
                     << endl;
                exit(EXIT_FAILURE);
            }

            for (uint chunkIndex = 0; chunkIndex < chunksCount; ++chunkIndex)
            {
                chunks_.push_back(dpct::device_pointer<T>(
                    reinterpret_cast<T *>(static_cast<char *>(basePtr_.get()) +
                                          chunkRawSize_ * chunkIndex)));
            }
        }

        ~DeviceMemoryPool()
        {
            try
            {
                dpct::free_device(basePtr_);
            }
            catch (std::system_error &e)
            {
                cout << "Pool memory allocation failed (" << e.what() << ")"
                     << endl;
                exit(EXIT_FAILURE);
            }
        }

        // Returns an address of the first available array
        // in the memory pool.
        dpct::device_pointer<T> get()
        {
            dpct::device_pointer<T> ptr(chunks_.back());
            chunks_.pop_back();

            return ptr;
        }

        // Pushes an address stored in "ptr" to the list
        // of available arrays of the memory pool.
        // It should be noted that it is user who is responsible for returning
        // the previously requested memory to the appropriate pool.
        inline void put(const dpct::device_pointer<T> &ptr)
        {
            chunks_.push_back(ptr);
        }

        uint totalFreeChunks() const
        {
            return chunks_.size();
        }

    private:
        uint chunkSize_, chunkRawSize_;
        dpct::device_pointer<void> basePtr_;

        list<dpct::device_pointer<T>> chunks_;
};

// Graph structure.
struct Graph
{
    Graph() {}

    Graph(uint verticesCount, uint edgesCount) :
        vertices(verticesCount),
        edges(edgesCount),
        weights(edgesCount)
    {}

    // This vector stores offsets for each vertex in "edges" and "weights"
    // vectors. For example:
    // "vertices[0]" is an index of the first outgoing edge of vertex #0,
    // "vertices[1]" is an index of the first outgoing edge of vertex #1, etc.
    vector<uint> vertices;

    // This vector stores indices of endpoints of the corresponding edges.
    // For example, "edges[vertices[0]]" is the first neighbouring vertex
    // of vertex #0.
    vector<uint> edges;

    // This vector stores weights of the corresponding edges.
    vector<float> weights;
};

// Simple segmentation tree class.
// Each level of the tree corresponds to the segmentation.
// See "Level" class for the details.
class Pyramid
{
    public:
        void addLevel(uint totalSuperNodes, uint totalNodes,
                      dpct::device_pointer<uint> superVerticesOffsets,
                      dpct::device_pointer<uint> verticesIDs)
        {
            levels_.push_back(Level(totalSuperNodes, totalNodes));
            levels_.back().buildFromDeviceData(superVerticesOffsets,
                                               verticesIDs);
        }

        uint levelsCount() const
        {
            return static_cast<uint>(levels_.size());
        }

        void dump(uint width, uint height) const
        {
            char filename[256], format[256];
            uint levelIndex = 0;

            uint requiredDigitsCount =
                static_cast<uint>(log10(static_cast<float>(levelsCount()))) +
                1;
            sprintf(format, "level_%%0%uu.ppm", requiredDigitsCount);

            for (LevelsIterator level = levels_.rbegin();
                 level != levels_.rend();
                 ++level, ++levelIndex)
            {

                sprintf(filename, format, levelIndex);
                dumpLevel(level, width, height, filename);
            }
        }

    private:
        // Level of the segmentation tree.
        class Level
        {
            public:
                Level(uint totalSuperNodes, uint totalNodes) :
                    superNodesOffsets_(totalSuperNodes), nodes_(totalNodes)
                {
                }

                void buildFromDeviceData(
                    dpct::device_pointer<uint> superVerticesOffsets,
                    dpct::device_pointer<uint> verticesIDs)
                {
                    DPCT_CHECK_ERROR(
                        dpct::get_default_queue()
                            .memcpy(&(superNodesOffsets_[0]),
                                    superVerticesOffsets.get(),
                                    sizeof(uint) * superNodesOffsets_.size())
                            .wait());

                    DPCT_CHECK_ERROR(
                        dpct::get_default_queue()
                            .memcpy(&(nodes_[0]), verticesIDs.get(),
                                    sizeof(uint) * nodes_.size())
                            .wait());
                }

            private:
                friend class Pyramid;

                // The pair of the following vectors describes the
                // relation between the consecutive levels.
                // Consider an example. Let the index of the current level be n.
                // Then nodes of level #(n-1) with indices stored in
                // "nodes[superNodesOffsets_[0]]",
                // "nodes[superNodesOffsets_[0] + 1]",
                // ...,
                // "nodes[superNodesOffsets_[1] - 1]"
                // correspond to vertex #0 of level #n. An so on.
                vector<uint> superNodesOffsets_;
                vector<uint> nodes_;
        };

        typedef list<Level>::const_reverse_iterator LevelsIterator;

        // Dumps level to the file "level_n.ppm" where n
        // is index of the level. Segments are drawn in random colors.
        void dumpLevel(LevelsIterator level,
                       uint width,
                       uint height,
                       const char *filename) const
        {
            deque< std::pair<uint, uint> > nodesQueue;

            uint totalSegments;

            {
                const vector<uint> &superNodesOffsets =
                    level->superNodesOffsets_;
                const vector<uint> &nodes =
                    level->nodes_;

                totalSegments = static_cast<uint>(superNodesOffsets.size());

                for (uint superNodeIndex = 0, nodeIndex = 0;
                     superNodeIndex < superNodesOffsets.size();
                     ++superNodeIndex)
                {

                    uint superNodeEnd =
                        superNodeIndex + 1 < superNodesOffsets.size() ?
                        superNodesOffsets[superNodeIndex + 1] :
                        static_cast<uint>(nodes.size());

                    for (; nodeIndex < superNodeEnd; ++nodeIndex)
                    {
                        nodesQueue.push_back(std::make_pair(nodes[nodeIndex],
                                                            superNodeIndex));
                    }
                }
            }

            ++level;

            while (level != levels_.rend())
            {
                uint superNodesCount = static_cast<uint>(nodesQueue.size());

                const vector<uint> &superNodesOffsets =
                    level->superNodesOffsets_;
                const vector<uint> &nodes =
                    level->nodes_;

                while (superNodesCount--)
                {
                    std::pair<uint, uint> currentNode = nodesQueue.front();
                    nodesQueue.pop_front();

                    uint superNodeBegin = superNodesOffsets[currentNode.first];

                    uint superNodeEnd =
                        currentNode.first + 1 < superNodesOffsets.size() ?
                        superNodesOffsets[currentNode.first + 1] :
                        static_cast<uint>(nodes.size());

                    for (uint nodeIndex = superNodeBegin;
                         nodeIndex < superNodeEnd;
                         ++nodeIndex)
                    {

                        nodesQueue.push_back(
                            std::make_pair(nodes[nodeIndex],
                                           currentNode.second));
                    }
                }

                ++level;
            }

            vector<uint> colors(3 * totalSegments);

            for (uint colorIndex = 0; colorIndex < totalSegments; ++colorIndex)
            {
                colors[colorIndex * 3    ] = myrand() % 256;
                colors[colorIndex * 3 + 1] = myrand() % 256;
                colors[colorIndex * 3 + 2] = myrand() % 256;
            }

            uchar *image = new uchar[width * height * 3];

            while (!nodesQueue.empty())
            {
                std::pair<uint, uint> currentNode = nodesQueue.front();
                nodesQueue.pop_front();

                uint pixelIndex = currentNode.first;
                uint pixelSegment = currentNode.second;

                image[pixelIndex * 3    ] = colors[pixelSegment * 3    ];
                image[pixelIndex * 3 + 1] = colors[pixelSegment * 3 + 1];
                image[pixelIndex * 3 + 2] = colors[pixelSegment * 3 + 2];
            }

            __savePPM(filename, image, width, height, 3);

            delete[] image;
        }

        list<Level> levels_;
};

// The class that encapsulates the main algorithm.
class SegmentationTreeBuilder
{
    public:
        SegmentationTreeBuilder():verticesCount_(0),edgesCount_(0)  {}

        ~SegmentationTreeBuilder() {}

        // Repeatedly invokes the step of the algorithm
        // until the limiting segmentation is found.
        // Returns time (in ms) spent on building the tree.
        float run(const Graph &graph, Pyramid &segmentations)
        {
            dpct::event_ptr start, stop;
            std::chrono::time_point<std::chrono::steady_clock> start_ct1;
            std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

            start = new sycl::event();
            stop = new sycl::event();

            start_ct1 = std::chrono::steady_clock::now();

            // Allocate required memory pools. We need just 4 types of arrays.
            MemoryPoolsCollection pools =
            {
                DeviceMemoryPool<uint>(
                    static_cast<uint>(graph.vertices.size()),
                    kUintVerticesPoolsRequired),
                DeviceMemoryPool<float>(
                    static_cast<uint>(graph.vertices.size()),
                    kFloatVerticesPoolsRequired),
                DeviceMemoryPool<uint>(
                    static_cast<uint>(graph.edges.size()),
                    kUintEdgesPoolsRequired),
                DeviceMemoryPool<float>(
                    static_cast<uint>(graph.edges.size()),
                    kFloatEdgesPoolsRequired)
            };
            // Initialize internal variables
            try
            {
                initalizeData(graph, pools);
            }
            catch (std::system_error &e)
            {
                cout << "Initialization failed (" << e.what() << ")" << endl;
                exit(EXIT_FAILURE);
            }

            // Run steps
            AlgorithmStatus status;

            try
            {
                do
                {
                    status = invokeStep(pools, segmentations);
                }
                while (status != ALGORITHM_FINISHED);
            }
            catch (std::system_error &e)
            {
                cout << "Algorithm failed (" << e.what() << ")" << endl;
                exit(EXIT_FAILURE);
            }

            stop_ct1 = std::chrono::steady_clock::now();

            float elapsedTime;
            elapsedTime =
                std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();

            return 0;//elapsedTime;
        }

    private:
        void printMemoryUsage()
        {
            size_t availableMemory, totalMemory, usedMemory;

            dpct::get_current_device().get_memory_info(availableMemory,
                                                       totalMemory);
            usedMemory = totalMemory - availableMemory;

            cout << "Device memory: used " << usedMemory
                 << " available " << availableMemory
                 << " total " << totalMemory << endl;
        }

        struct MemoryPoolsCollection
        {
            DeviceMemoryPool<uint> uintVertices;
            DeviceMemoryPool<float> floatVertices;
            DeviceMemoryPool<uint> uintEdges;
            DeviceMemoryPool<float> floatEdges;
        };

        static const uint kUintVerticesPoolsRequired = 8;
        static const uint kFloatVerticesPoolsRequired = 3;
        static const uint kUintEdgesPoolsRequired = 8;
        static const uint kFloatEdgesPoolsRequired = 4;

        void initalizeData(const Graph &graph, MemoryPoolsCollection &pools)
        {
            // Get memory for the internal variables
            verticesCount_ = static_cast<uint>(graph.vertices.size());
            edgesCount_ = static_cast<uint>(graph.edges.size());

            dVertices_ = pools.uintVertices.get();
            dEdges_ = pools.uintEdges.get();
            dWeights_ = pools.floatEdges.get();

            dOutputEdgesFlags_ = pools.uintEdges.get();

            // Copy graph to the device memory
            DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(dVertices_.get(), &(graph.vertices[0]),
                            sizeof(uint) * verticesCount_)
                    .wait());
            DPCT_CHECK_ERROR(dpct::get_default_queue()
                                     .memcpy(dEdges_.get(), &(graph.edges[0]),
                                             sizeof(uint) * edgesCount_)
                                     .wait());
            DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(dWeights_.get(), &(graph.weights[0]),
                            sizeof(float) * edgesCount_)
                    .wait());

            std::fill(oneapi::dpl::execution::make_device_policy(
                          dpct::get_default_queue()),
                      dOutputEdgesFlags_, dOutputEdgesFlags_ + edgesCount_, 0);
        }

        static const uint kMaxThreadsPerBlock = 256;

        // Calculates grid parameters of the consecutive kernel calls
        // based on the number of elements in the array.
        void calculateThreadsDistribution(uint totalElements,
                                          uint &blocksCount,
                                          uint &threadsPerBlockCount)
        {
            if (totalElements > kMaxThreadsPerBlock)
            {
                blocksCount =
                    (totalElements + kMaxThreadsPerBlock - 1) /
                    kMaxThreadsPerBlock;

                threadsPerBlockCount = kMaxThreadsPerBlock;
            }
            else
            {
                blocksCount = 1;
                threadsPerBlockCount = totalElements;
            }
        }

        enum AlgorithmStatus { ALGORITHM_NOT_FINISHED, ALGORITHM_FINISHED };

        AlgorithmStatus invokeStep(MemoryPoolsCollection &pools,
                                   Pyramid &segmentations)
        {
            uint blocksCount, threadsPerBlockCount;

            calculateThreadsDistribution(edgesCount_,
                                         blocksCount,
                                         threadsPerBlockCount);
            sycl::range<3> gridDimsForEdges(1, 1, blocksCount);
            sycl::range<3> blockDimsForEdges(1, 1, threadsPerBlockCount);

            calculateThreadsDistribution(verticesCount_,
                                         blocksCount,
                                         threadsPerBlockCount);
            sycl::range<3> gridDimsForVertices(1, 1, blocksCount);
            sycl::range<3> blockDimsForVertices(1, 1, threadsPerBlockCount);

            dpct::device_pointer<uint> dEdgesFlags = pools.uintEdges.get();

            std::fill(oneapi::dpl::execution::make_device_policy(
                          dpct::get_default_queue()),
                      dEdgesFlags, dEdgesFlags + edgesCount_, 0);

            // Mark the first edge for each vertex in "dEdgesFlags"
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                        auto dVertices__get_ct0 =
                                            dVertices_.get();
                                        auto dEdgesFlags_get_ct1 =
                                            dEdgesFlags.get();
                                        auto verticesCount__ct2 =
                                            verticesCount_;

                                        cgh.parallel_for(
                                            sycl::nd_range<3>(
                                                gridDimsForVertices *
                                                    blockDimsForVertices,
                                                blockDimsForVertices),
                                            [=](sycl::nd_item<3> item_ct1) {
                                                        markSegments(
                                                            dVertices__get_ct0,
                                                            dEdgesFlags_get_ct1,
                                                            verticesCount__ct2,
                                                            item_ct1);
                                            });
                            });
            getLastCudaError("markSegments launch failed.");

            // Now find minimum edges for each vertex.
            dpct::device_pointer<uint> dMinScannedEdges = pools.uintEdges.get();
            dpct::device_pointer<float> dMinScannedWeights = pools.floatEdges.get();

            oneapi::dpl::inclusive_scan_by_segment(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                dEdgesFlags, dEdgesFlags + edgesCount_,
                oneapi::dpl::make_zip_iterator(
                    std::make_tuple(dWeights_, dEdges_)),
                oneapi::dpl::make_zip_iterator(
                    std::make_tuple(dMinScannedWeights, dMinScannedEdges)),
                std::greater_equal<uint>(),
                oneapi::dpl::minimum<std::tuple<float, uint>>());

            // To make things clear.
            // Let "dEdgesFlags" denote groups of edges that
            // correspond to the same vertices. Then the last edge of each group
            // (in "dMinScannedEdges" and "dMinScannedWeights") is now minimal.

            // Calculate a successor vertex for each vertex. A successor of the
            // vertex v is a neighbouring vertex connected to v
            // by the minimal edge.
            dpct::device_pointer<uint> dSuccessors = pools.uintVertices.get();

                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dVertices__get_ct0 = dVertices_.get();
                                    auto dMinScannedEdges_get_ct1 =
                                        dMinScannedEdges.get();
                                    auto dSuccessors_get_ct2 =
                                        dSuccessors.get();
                                    auto verticesCount__ct3 = verticesCount_;
                                    auto edgesCount__ct4 = edgesCount_;

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(
                                            gridDimsForVertices *
                                                blockDimsForVertices,
                                            blockDimsForVertices),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    getSuccessors(
                                                        dVertices__get_ct0,
                                                        dMinScannedEdges_get_ct1,
                                                        dSuccessors_get_ct2,
                                                        verticesCount__ct3,
                                                        edgesCount__ct4,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("getSuccessors launch failed.");

            pools.uintEdges.put(dMinScannedEdges);
            pools.floatEdges.put(dMinScannedWeights);

            // Remove cyclic successor dependencies. Note that there can be only
            // two vertices in a cycle. See [1] for details.
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                        auto dSuccessors_get_ct0 =
                                            dSuccessors.get();
                                        auto verticesCount__ct1 =
                                            verticesCount_;

                                        cgh.parallel_for(
                                            sycl::nd_range<3>(
                                                gridDimsForVertices *
                                                    blockDimsForVertices,
                                                blockDimsForVertices),
                                            [=](sycl::nd_item<3> item_ct1) {
                                                        removeCycles(
                                                            dSuccessors_get_ct0,
                                                            verticesCount__ct1,
                                                            item_ct1);
                                            });
                            });
            getLastCudaError("removeCycles launch failed.");

            // Build up an array of startpoints for edges. As already stated,
            // each group of edges denoted by "dEdgesFlags"
            // has the same startpoint.
            dpct::device_pointer<uint> dStartpoints = pools.uintEdges.get();

            oneapi::dpl::inclusive_scan(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                dEdgesFlags, dEdgesFlags + edgesCount_, dStartpoints);

                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dStartpoints_get_ct0 =
                                        dStartpoints.get();
                                    auto edgesCount__ct2 = edgesCount_;

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(gridDimsForEdges *
                                                              blockDimsForEdges,
                                                          blockDimsForEdges),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    addScalar(
                                                        dStartpoints_get_ct0,
                                                        -1, edgesCount__ct2,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("addScalar launch failed.");

            // Shrink the chains of successors. New successors will eventually
            // represent superpixels of the new level.
            dpct::device_pointer<uint> dRepresentatives = pools.uintVertices.get();

                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dSuccessors_get_ct0 =
                                        dSuccessors.get();
                                    auto dRepresentatives_get_ct1 =
                                        dRepresentatives.get();
                                    auto verticesCount__ct2 = verticesCount_;

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(
                                            gridDimsForVertices *
                                                blockDimsForVertices,
                                            blockDimsForVertices),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    getRepresentatives(
                                                        dSuccessors_get_ct0,
                                                        dRepresentatives_get_ct1,
                                                        verticesCount__ct2,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("getRepresentatives launch failed.");

            swap(dSuccessors, dRepresentatives);

            pools.uintVertices.put(dRepresentatives);

            // Group vertices by successors' indices.
            dpct::device_pointer<uint> dClusteredVerticesIDs =
                pools.uintVertices.get();

            dpct::iota(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       dClusteredVerticesIDs,
                       dClusteredVerticesIDs + verticesCount_);

            oneapi::dpl::sort(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                oneapi::dpl::make_zip_iterator(std::make_tuple(
                    dpct::device_pointer<uint>(dSuccessors),
                    dpct::device_pointer<uint>(dClusteredVerticesIDs))),
                oneapi::dpl::make_zip_iterator(std::make_tuple(
                    dpct::device_pointer<uint>(dSuccessors + verticesCount_),
                    dpct::device_pointer<uint>(dClusteredVerticesIDs +
                                               verticesCount_))));

            // Mark those groups.
            dpct::device_pointer<uint> dVerticesFlags_ = pools.uintVertices.get();

            std::fill(oneapi::dpl::execution::make_device_policy(
                          dpct::get_default_queue()),
                      dVerticesFlags_, dVerticesFlags_ + verticesCount_, 0);

            oneapi::dpl::adjacent_difference(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                dSuccessors, dSuccessors + verticesCount_, dVerticesFlags_,
                std::not_equal_to<uint>());

            dpct::get_default_queue()
                .memset((void *)dVerticesFlags_.get(), 0, sizeof(uint))
                .wait();

            // Assign new indices to the successors (the indices of vertices
            // at the new level).
            dpct::device_pointer<uint> dNewVerticesIDs_ = pools.uintVertices.get();

            oneapi::dpl::inclusive_scan(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                dVerticesFlags_, dVerticesFlags_ + verticesCount_,
                dNewVerticesIDs_);

            pools.uintVertices.put(dVerticesFlags_);

            // Now we can calculate number of resulting superpixels easily.
            uint newVerticesCount;
            DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(&newVerticesCount,
                            (dNewVerticesIDs_ + verticesCount_ - 1).get(),
                            sizeof(uint))
                    .wait());
            ++newVerticesCount;

            // There are two special cases when we can stop our algorithm:
            // 1) number of vertices in the graph remained unchanged;
            // 2) only one vertex remains.
            if (newVerticesCount == verticesCount_)
            {
                return ALGORITHM_FINISHED;
            }
            else if (newVerticesCount == 1)
            {
                dpct::device_pointer<uint> dDummyVerticesOffsets =
                    pools.uintVertices.get();

                dpct::get_default_queue()
                    .memset((void *)dDummyVerticesOffsets.get(), 0,
                            sizeof(uint))
                    .wait();

                dpct::device_pointer<uint> dDummyVerticesIDs =
                    pools.uintVertices.get();

                dpct::iota(oneapi::dpl::execution::make_device_policy(
                               dpct::get_default_queue()),
                           dDummyVerticesIDs,
                           dDummyVerticesIDs + verticesCount_);

                segmentations.addLevel(1,
                                       verticesCount_,
                                       dDummyVerticesOffsets,
                                       dDummyVerticesIDs);

                return ALGORITHM_FINISHED;
            }

            // Calculate how old vertices IDs map to new vertices IDs.
            dpct::device_pointer<uint> dVerticesMapping = pools.uintVertices.get();

                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dClusteredVerticesIDs_get_ct0 =
                                        dClusteredVerticesIDs.get();
                                    auto dNewVerticesIDs__get_ct1 =
                                        dNewVerticesIDs_.get();
                                    auto dVerticesMapping_get_ct2 =
                                        dVerticesMapping.get();
                                    auto verticesCount__ct3 = verticesCount_;

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(
                                            gridDimsForVertices *
                                                blockDimsForVertices,
                                            blockDimsForVertices),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    getVerticesMapping(
                                                        dClusteredVerticesIDs_get_ct0,
                                                        dNewVerticesIDs__get_ct1,
                                                        dVerticesMapping_get_ct2,
                                                        verticesCount__ct3,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("getVerticesMapping launch failed.");

            pools.uintVertices.put(dNewVerticesIDs_);
            pools.uintVertices.put(dClusteredVerticesIDs);
            pools.uintVertices.put(dSuccessors);

            // Invalidate self-loops in the reduced graph (the graph
            // produced by merging all old vertices that have
            // the same successor).
                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dStartpoints_get_ct0 =
                                        dStartpoints.get();
                                    auto dVerticesMapping_get_ct1 =
                                        dVerticesMapping.get();
                                    auto dEdges__get_ct2 = dEdges_.get();
                                    auto edgesCount__ct3 = edgesCount_;

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(gridDimsForEdges *
                                                              blockDimsForEdges,
                                                          blockDimsForEdges),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    invalidateLoops(
                                                        dStartpoints_get_ct0,
                                                        dVerticesMapping_get_ct1,
                                                        dEdges__get_ct2,
                                                        edgesCount__ct3,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("invalidateLoops launch failed.");

            // Calculate various information about the surviving
            // (new startpoints IDs and IDs of edges) and
            // non-surviving/contracted edges (their weights).
            dpct::device_pointer<uint> dNewStartpoints = pools.uintEdges.get();
            dpct::device_pointer<uint> dSurvivedEdgesIDs = pools.uintEdges.get();

                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dStartpoints_get_ct0 =
                                        dStartpoints.get();
                                    auto dVerticesMapping_get_ct1 =
                                        dVerticesMapping.get();
                                    auto dEdges__get_ct2 = dEdges_.get();
                                    auto dWeights__get_ct3 = dWeights_.get();
                                    auto dNewStartpoints_get_ct4 =
                                        dNewStartpoints.get();
                                    auto dSurvivedEdgesIDs_get_ct5 =
                                        dSurvivedEdgesIDs.get();
                                    auto edgesCount__ct6 = edgesCount_;

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(gridDimsForEdges *
                                                              blockDimsForEdges,
                                                          blockDimsForEdges),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    calculateEdgesInfo(
                                                        dStartpoints_get_ct0,
                                                        dVerticesMapping_get_ct1,
                                                        dEdges__get_ct2,
                                                        dWeights__get_ct3,
                                                        dNewStartpoints_get_ct4,
                                                        dSurvivedEdgesIDs_get_ct5,
                                                        edgesCount__ct6,
                                                        newVerticesCount,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("calculateEdgesInfo launch failed.");

            pools.uintEdges.put(dStartpoints);

            // Group that information by the new startpoints IDs.
            // Keep in mind that we want to build new (reduced) graph and apply
            // the step of the algorithm to that one. Hence we need to
            // preserve the structure of the original graph: neighbours and
            // weights should be grouped by vertex.
            oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(
                                  dpct::get_default_queue()),
                              oneapi::dpl::make_zip_iterator(std::make_tuple(
                                  dNewStartpoints, dSurvivedEdgesIDs)),
                              oneapi::dpl::make_zip_iterator(std::make_tuple(
                                  dNewStartpoints + edgesCount_,
                                  dSurvivedEdgesIDs + edgesCount_)));

            // Find the group of contracted edges.
            uint *invalidEdgesPtr =
                oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(
                                         dpct::get_default_queue()),
                                     dNewStartpoints,
                                     dNewStartpoints + edgesCount_,
                                     IsGreaterEqualThan<uint>(newVerticesCount))
                    .get();

            // Calculate how many edges there are in the reduced graph.
            uint validEdgesCount =
                static_cast<uint>(invalidEdgesPtr - dNewStartpoints.get());

            // Mark groups of edges corresponding to the same vertex in the
            // reduced graph.
            oneapi::dpl::adjacent_difference(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                dNewStartpoints, dNewStartpoints + edgesCount_, dEdgesFlags,
                std::not_equal_to<uint>());

            dpct::get_default_queue()
                .memset((void *)dEdgesFlags.get(), 0, sizeof(uint))
                .wait();
            dpct::get_default_queue()
                .memset((void *)dEdgesFlags.get(), 1, 1)
                .wait();

            pools.uintEdges.put(dNewStartpoints);

            // Now we are able to build the reduced graph. See "Graph"
            // class for the details on the graph's internal structure.

            // Calculate vertices' offsets for the reduced graph.
            dpct::copy_if(oneapi::dpl::execution::seq,
                          dpct::make_counting_iterator(0U),
                          dpct::make_counting_iterator(validEdgesCount),
                          dEdgesFlags, dVertices_, oneapi::dpl::identity())
                .get();

            pools.uintEdges.put(dEdgesFlags);

            // Build up a neighbourhood for each vertex in the reduced graph
            // (this includes recalculating edges' weights).
            calculateThreadsDistribution(validEdgesCount,
                                         blocksCount,
                                         threadsPerBlockCount);
            sycl::range<3> newGridDimsForEdges(1, 1, blocksCount);
            sycl::range<3> newBlockDimsForEdges(1, 1, threadsPerBlockCount);

            dpct::device_pointer<uint> dNewEdges = pools.uintEdges.get();
            dpct::device_pointer<float> dNewWeights = pools.floatEdges.get();

                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                                    auto dSurvivedEdgesIDs_get_ct0 =
                                        dSurvivedEdgesIDs.get();
                                    auto dVerticesMapping_get_ct1 =
                                        dVerticesMapping.get();
                                    auto dEdges__get_ct2 = dEdges_.get();
                                    auto dWeights__get_ct3 = dWeights_.get();
                                    auto dNewEdges_get_ct4 = dNewEdges.get();
                                    auto dNewWeights_get_ct5 =
                                        dNewWeights.get();

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(
                                            newGridDimsForEdges *
                                                newBlockDimsForEdges,
                                            newBlockDimsForEdges),
                                        [=](sycl::nd_item<3> item_ct1) {
                                                    makeNewEdges(
                                                        dSurvivedEdgesIDs_get_ct0,
                                                        dVerticesMapping_get_ct1,
                                                        dEdges__get_ct2,
                                                        dWeights__get_ct3,
                                                        dNewEdges_get_ct4,
                                                        dNewWeights_get_ct5,
                                                        validEdgesCount,
                                                        item_ct1);
                                        });
                        });
            getLastCudaError("makeNewEdges launch failed.");

            swap(dEdges_, dNewEdges);
            swap(dWeights_, dNewWeights);

            pools.uintEdges.put(dNewEdges);
            pools.floatEdges.put(dNewWeights);

            pools.uintEdges.put(dSurvivedEdgesIDs);

            // The graph's reconstruction is now finished.

            // Build new level of the segmentation tree. It is a trivial task
            // as we already have "dVerticesMapping" that contains all
            // sufficient information about the vertices' transformations.
            dpct::device_pointer<uint> dVerticesIDs = pools.uintVertices.get();
            dpct::device_pointer<uint> dNewVerticesOffsets =
                pools.uintVertices.get();

            dpct::iota(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       dVerticesIDs, dVerticesIDs + verticesCount_);

            dpct::sort(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       dVerticesMapping, dVerticesMapping + verticesCount_,
                       dVerticesIDs);

            dpct::unique_copy(
                oneapi::dpl::execution::make_device_policy(
                    dpct::get_default_queue()),
                dVerticesMapping, dVerticesMapping + verticesCount_,
                dpct::make_counting_iterator(0),
                oneapi::dpl::discard_iterator(), dNewVerticesOffsets);

            segmentations.addLevel(newVerticesCount,
                                   verticesCount_,
                                   dNewVerticesOffsets,
                                   dVerticesIDs);

            pools.uintVertices.put(dVerticesIDs);
            pools.uintVertices.put(dNewVerticesOffsets);
            pools.uintVertices.put(dVerticesMapping);

            // We can now safely set new counts for vertices and edges.
            verticesCount_ = newVerticesCount;
            edgesCount_ = validEdgesCount;

            return ALGORITHM_NOT_FINISHED;
        }

        uint verticesCount_;
        uint edgesCount_;

        dpct::device_pointer<uint> dVertices_;
        dpct::device_pointer<uint> dEdges_;
        dpct::device_pointer<float> dWeights_;

        dpct::device_pointer<uint> dOutputEdgesFlags_;
};

// Loads PPM image.
int loadImage(const char *filename, const char *executablePath,
              vector<sycl::uchar3> &data, uint &width, uint &height)
{
    
    const char *imagePath = sdkFindFilePath(filename, executablePath);

    if (imagePath == NULL)
    {
        return -1;
    }

    uchar *dataHandle = NULL;
    unsigned int channels;

    if (!__loadPPM(imagePath, &dataHandle, &width, &height, &channels))
    {
        return -1;
    }
    
    data.assign(reinterpret_cast<sycl::uchar3 *>(&dataHandle),
                reinterpret_cast<sycl::uchar3 *>(&dataHandle) + width * height);

    free(reinterpret_cast<void *>(dataHandle));

    return 0;
}

inline float distance(const sycl::uchar3 &first, const sycl::uchar3 &second)
{
    int dx = static_cast<int>(first.x()) - static_cast<int>(second.x());
    int dy = static_cast<int>(first.y()) - static_cast<int>(second.y());
    int dz = static_cast<int>(first.z()) - static_cast<int>(second.z());

    uint sqrResult = dx * dx + dy * dy + dz * dz;

    return sqrt(static_cast<float>(sqrResult));
}

// Builds a net-graph for the image with 4-connected pixels.
void buildGraph(const vector<sycl::uchar3> &image, uint width, uint height,
                Graph &graph)
{
    uint totalNodes = static_cast<uint>(image.size());

    graph.vertices.resize(totalNodes);
    graph.edges.reserve(4 * totalNodes - 2 * (width + height));
    // graph.weights.reserve(graph.edges.size());

    uint edgesProcessed = 0;

    // for (uint y = 0; y < height; ++y)
    // {
    //     for (uint x = 0; x < width; ++x)
    //     {
    //         uint nodeIndex = y * width + x;
    //         const sycl::uchar3 &centerPixel = image[nodeIndex];

    //         graph.vertices[nodeIndex] = edgesProcessed;

    //         if (y > 0)
    //         {
    //             uint lowerNodeIndex = (y - 1) * width + x;
    //             const sycl::uchar3 &lowerPixel = image[lowerNodeIndex];

    //             graph.edges.push_back(lowerNodeIndex);
    //             graph.weights.push_back(distance(centerPixel, lowerPixel));

    //             ++edgesProcessed;
    //         }

    //         if (y + 1 < height)
    //         {
    //             uint upperNodeIndex = (y + 1) * width + x;
    //             const sycl::uchar3 &upperPixel = image[upperNodeIndex];

    //             graph.edges.push_back(upperNodeIndex);
    //             graph.weights.push_back(distance(centerPixel, upperPixel));

    //             ++edgesProcessed;
    //         }

    //         if (x > 0)
    //         {
    //             uint leftNodeIndex = y * width + x - 1;
    //             const sycl::uchar3 &leftPixel = image[leftNodeIndex];

    //             graph.edges.push_back(leftNodeIndex);
    //             graph.weights.push_back(distance(centerPixel, leftPixel));

    //             ++edgesProcessed;
    //         }

    //         if (x + 1 < width)
    //         {
    //             uint rightNodeIndex = y * width + x + 1;
    //             const sycl::uchar3 &rightPixel = image[rightNodeIndex];

    //             graph.edges.push_back(rightNodeIndex);
    //             graph.weights.push_back(distance(centerPixel, rightPixel));

    //             ++edgesProcessed;
    //         }
    //     }
    // }
}

static char *kDefaultImageName = (char*)"test.ppm";

int main(int argc, char **argv)
{
    vector<sycl::uchar3> image;
    uint imageWidth, imageHeight;
    char *imageName;

    printf("%s Starting...\n\n", argv[0]);

    imageName = (char *)kDefaultImageName;

    if (checkCmdLineFlag(argc, (const char **) argv, "file"))
    {
        getCmdLineArgumentString(argc,
                                 (const char **) argv,
                                 "file",
                                 &imageName);
    } 
    int temp = loadImage(imageName, argv[0], image, imageWidth, imageHeight);
    if (temp != 0)
    {
        printf("Failed to open <%s>, program exit...\n", imageName);
        exit(EXIT_FAILURE);
    }

    findCudaDevice(argc, (const char **)argv);
    Graph graph;
    buildGraph(image, imageWidth, imageHeight, graph);

    Pyramid segmentations;

    cout << "* Building segmentation tree... ";
    cout.flush();

    SegmentationTreeBuilder algo;
    // float elapsedTime = algo.run(graph, segmentations);

    // cout << "done in " << elapsedTime << " (ms)" << endl;

    cout << "* Dumping levels for each tree..." << endl << endl;

    segmentations.dump(imageWidth, imageHeight);

    bool bResults[2];

    bResults[0] = sdkComparePPM("level_00.ppm",
                                sdkFindFilePath("ref_00.ppm", argv[0]),
                                5.0f,
                                0.15f,
                                false);
    bResults[1] = sdkComparePPM("level_09.ppm",
                                sdkFindFilePath("ref_09.ppm", argv[0]),
                                5.0f,
                                0.15f,
                                false);

    exit((bResults[0] && bResults[1]) ? EXIT_SUCCESS : EXIT_FAILURE);
}

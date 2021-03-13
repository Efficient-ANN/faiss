/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/IMIBase.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>

namespace faiss {
namespace gpu {

IMIBase::DeviceIVFList::DeviceIVFList(GpuResources *res, const AllocInfo &info)
    : data(res, info), numVecs(0) {}

IMIBase::IMIBase(GpuResources *resources, MultiIndex2 *quantizer,
                 bool interleavedLayout, IndicesOptions indicesOptions,
                 MemorySpace space)
    : resources_(resources), quantizer_(quantizer), dim_(quantizer->getDim()),
      numLists_(quantizer->getSize()), interleavedLayout_(interleavedLayout),
      indicesOptions_(indicesOptions), space_(space),
      deviceListDataPointers_(
          resources, makeDevAlloc(AllocType::Other,
                                  resources->getDefaultStreamCurrentDevice())),
      deviceListIndexPointers_(
          resources, makeDevAlloc(AllocType::Other,
                                  resources->getDefaultStreamCurrentDevice())),
      deviceListIndexPointersIdxT_(
          resources, makeDevAlloc(AllocType::Other,
                                  resources->getDefaultStreamCurrentDevice())),
      deviceListLengths_(
          resources, makeDevAlloc(AllocType::Other,
                                  resources->getDefaultStreamCurrentDevice())),
      maxListLength_(0) {
  reset();
}

IMIBase::~IMIBase() {}

void IMIBase::reserveMemory(size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  auto vecsPerList = numVecs / deviceListData_.size();
  if (vecsPerList < 1) {
    return;
  }

  auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);

  for (auto &list : deviceListData_) {
    list->data.reserve(bytesPerDataList, stream);
  }

  if ((indicesOptions_ == INDICES_32_BIT) ||
      (indicesOptions_ == INDICES_64_BIT)) {
    // Reserve for index lists as well
    size_t bytesPerIndexList = vecsPerList * (indicesOptions_ == INDICES_32_BIT
                                                  ? sizeof(int)
                                                  : sizeof(Index::idx_t));

    for (auto &list : deviceListIndices_) {
      list->data.reserve(bytesPerIndexList, stream);
    }
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);
}

void IMIBase::reserveMemory(
    const std::unordered_map<int, int> *expectedNumAddsPerList) {
  if (!expectedNumAddsPerList || expectedNumAddsPerList->empty()) {
    return;
  }

  auto stream = resources_->getDefaultStreamCurrentDevice();

  for (auto &expectedNumAdds : *expectedNumAddsPerList) {
    size_t bytesPerDataList =
        getGpuVectorsEncodingSize_(expectedNumAdds.second);
    deviceListData_[expectedNumAdds.first]->data.reserve(bytesPerDataList,
                                                         stream);
  }

  size_t bytesPerIndexList;
  for (auto &expectedNumAdds : *expectedNumAddsPerList) {
    if ((indicesOptions_ == INDICES_32_BIT) ||
        (indicesOptions_ == INDICES_64_BIT)) {
      bytesPerIndexList =
          expectedNumAdds.second * (indicesOptions_ == INDICES_32_BIT
                                        ? sizeof(int)
                                        : sizeof(Index::idx_t));
    }
    deviceListIndices_[expectedNumAdds.first]->data.reserve(bytesPerIndexList,
                                                            stream);
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);
}

void IMIBase::reset() {
  deviceListData_.clear();
  deviceListIndices_.clear();
  deviceListDataPointers_.clear();
  deviceListIndexPointers_.clear();
  deviceListIndexPointersIdxT_.clear();
  deviceListLengths_.clear();
  listOffsetToUserIndex_.clear();

  auto info = AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_,
                        resources_->getDefaultStreamCurrentDevice());

  for (size_t i = 0; i < numLists_; ++i) {
    deviceListData_.emplace_back(
        std::unique_ptr<DeviceIVFList>(new DeviceIVFList(resources_, info)));

    deviceListIndices_.emplace_back(
        std::unique_ptr<DeviceIVFList>(new DeviceIVFList(resources_, info)));

    listOffsetToUserIndex_.emplace_back(std::vector<Index::idx_t>());
  }

  auto stream = resources_->getDefaultStreamCurrentDevice();

  deviceListDataPointers_.reserve(numLists_, stream);
  deviceListDataPointers_.resize(numLists_, stream);
  DeviceTensor<uint8_t *, 1, true> deviceListDataPointersTensor(
      deviceListDataPointers_.data(), {(int)deviceListDataPointers_.size()});
  thrust::fill(thrust::cuda::par.on(stream),
               deviceListDataPointersTensor.data(),
               deviceListDataPointersTensor.end(), nullptr);

  if (indicesOptions_ == INDICES_64_BIT) {
    deviceListIndexPointersIdxT_.reserve(numLists_, stream);
    deviceListIndexPointersIdxT_.resize(numLists_, stream);
    DeviceTensor<Index::idx_t *, 1, true> deviceListIndexPointersTensor(
        deviceListIndexPointersIdxT_.data(),
        {(int)deviceListIndexPointersIdxT_.size()});
    thrust::fill(thrust::cuda::par.on(stream),
                 deviceListIndexPointersTensor.data(),
                 deviceListIndexPointersTensor.end(), nullptr);
  } else {
    deviceListIndexPointers_.reserve(numLists_, stream);
    deviceListIndexPointers_.resize(numLists_, stream);
    DeviceTensor<int *, 1, true> deviceListIndexPointersTensor(
        deviceListIndexPointers_.data(),
        {(int)deviceListIndexPointers_.size()});
    thrust::fill(thrust::cuda::par.on(stream),
                 deviceListIndexPointersTensor.data(),
                 deviceListIndexPointersTensor.end(), nullptr);
  }

  deviceListLengths_.reserve(numLists_, stream);
  deviceListLengths_.resize(numLists_, stream);
  DeviceTensor<int, 1, true> deviceListLengthsTensor(
      deviceListLengths_.data(), {(int)deviceListLengths_.size()});
  thrust::fill(thrust::cuda::par.on(stream), deviceListLengthsTensor.data(),
               deviceListLengthsTensor.end(), 0);

  maxListLength_ = 0;
}

int IMIBase::getMaxListLength() const { return maxListLength_; }

int IMIBase::getDim() const { return dim_; }

size_t IMIBase::reclaimMemory() {
  // Reclaim all unused memory exactly
  return reclaimMemory_(true);
}

size_t IMIBase::reclaimMemory_(bool exact) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  size_t totalReclaimed = 0;

  for (int i = 0; i < deviceListData_.size(); ++i) {
    auto &data = deviceListData_[i]->data;
    totalReclaimed += data.reclaim(exact, stream);

    deviceListDataPointers_.data()[i] = data.data();
  }

  if (indicesOptions_ == INDICES_64_BIT) {
    for (int i = 0; i < deviceListIndices_.size(); ++i) {
      auto &indices = deviceListIndices_[i]->data;
      totalReclaimed += indices.reclaim(exact, stream);

      deviceListIndexPointersIdxT_.data()[i] = (Index::idx_t *)indices.data();
    }
  } else {
    for (int i = 0; i < deviceListIndices_.size(); ++i) {
      auto &indices = deviceListIndices_[i]->data;
      totalReclaimed += indices.reclaim(exact, stream);

      deviceListIndexPointers_.data()[i] = (int *)indices.data();
    }
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);

  return totalReclaimed;
} // namespace gpu

void IMIBase::updateDeviceListInfo_(cudaStream_t stream) {
  std::vector<int> listIds(deviceListData_.size());
  for (int i = 0; i < deviceListData_.size(); ++i) {
    listIds[i] = i;
  }

  updateDeviceListInfo_(listIds, stream);
}

void IMIBase::updateDeviceListInfo_(const std::vector<int> &listIds,
                                    cudaStream_t stream) {
  HostTensor<int, 1, true> hostListsToUpdate({(int)listIds.size()});
  HostTensor<int, 1, true> hostNewListLength({(int)listIds.size()});
  HostTensor<uint8_t *, 1, true> hostNewDataPointers({(int)listIds.size()});

  for (int i = 0; i < listIds.size(); ++i) {
    auto listId = listIds[i];
    auto &data = deviceListData_[listId];

    hostListsToUpdate[i] = listId;
    hostNewListLength[i] = data->numVecs;
    hostNewDataPointers[i] = data->data.data();
  }

  // Copy the above update sets to the GPU
  DeviceTensor<int, 1, true> listsToUpdate(
      resources_, makeTempAlloc(AllocType::Other, stream), hostListsToUpdate);
  DeviceTensor<int, 1, true> newListLength(
      resources_, makeTempAlloc(AllocType::Other, stream), hostNewListLength);
  DeviceTensor<uint8_t *, 1, true> newDataPointers(
      resources_, makeTempAlloc(AllocType::Other, stream), hostNewDataPointers);

  DeviceTensor<uint8_t *, 1, true> deviceListDataPointersTensor(
      deviceListDataPointers_.data(), {(int)deviceListDataPointers_.size()});
  DeviceTensor<int, 1, true> deviceListLengthsTensor(
      deviceListLengths_.data(), {(int)deviceListLengths_.size()});

  if (indicesOptions_ == INDICES_64_BIT) {
    HostTensor<Index::idx_t *, 1, true> hostNewIndexPointers(
        {(int)listIds.size()});

    for (int i = 0; i < listIds.size(); ++i) {
      auto listId = listIds[i];
      auto &indices = deviceListIndices_[listId];
      hostNewIndexPointers[i] = (Index::idx_t *)indices->data.data();
    }

    DeviceTensor<Index::idx_t *, 1, true> newIndexPointers(
        resources_, makeTempAlloc(AllocType::Other, stream),
        hostNewIndexPointers);

    DeviceTensor<Index::idx_t *, 1, true> deviceListIndexPointersTensor(
        deviceListIndexPointersIdxT_.data(),
        {(int)deviceListIndexPointersIdxT_.size()});
    // Update all pointers to the lists on the device that may have
    // changed
    runUpdateListPointers(listsToUpdate, newListLength, newDataPointers,
                          newIndexPointers, deviceListLengthsTensor,
                          deviceListDataPointersTensor,
                          deviceListIndexPointersTensor, stream);
  } else {
    HostTensor<int *, 1, true> hostNewIndexPointers({(int)listIds.size()});

    for (int i = 0; i < listIds.size(); ++i) {
      auto listId = listIds[i];
      auto &indices = deviceListIndices_[listId];
      hostNewIndexPointers[i] = (int *)indices->data.data();
    }

    DeviceTensor<int *, 1, true> newIndexPointers(
        resources_, makeTempAlloc(AllocType::Other, stream),
        hostNewIndexPointers);

    DeviceTensor<int *, 1, true> deviceListIndexPointersTensor(
        deviceListIndexPointers_.data(),
        {(int)deviceListIndexPointers_.size()});

    // Update all pointers to the lists on the device that may have
    // changed
    runUpdateListPointers(listsToUpdate, newListLength, newDataPointers,
                          newIndexPointers, deviceListLengthsTensor,
                          deviceListDataPointersTensor,
                          deviceListIndexPointersTensor, stream);
  }
}

size_t IMIBase::getNumLists() const { return numLists_; }

int IMIBase::getListLength(int listId) const {
  FAISS_THROW_IF_NOT_FMT(listId < numLists_,
                         "IVF list %d is out of bounds (%d lists total)",
                         listId, numLists_);
  FAISS_ASSERT(listId < deviceListLengths_.size());
  FAISS_ASSERT(listId < deviceListData_.size());

  // LHS is the GPU resident value, RHS is the CPU resident value
  FAISS_ASSERT(deviceListLengths_.data()[listId] ==
               deviceListData_[listId]->numVecs);

  return deviceListData_[listId]->numVecs;
}

int IMIBase::getAllListsLength() const {
  int length = 0;
  for (int i = 0; i < deviceListLengths_.size(); i++) {
    length += getListLength(i);
  }
  return length;
}

std::vector<Index::idx_t> IMIBase::getListIndices(int listId) const {
  FAISS_THROW_IF_NOT_FMT(listId < numLists_,
                         "IVF list %d is out of bounds (%d lists total)",
                         listId, numLists_);
  FAISS_ASSERT(listId < deviceListData_.size());
  FAISS_ASSERT(listId < deviceListLengths_.size());

  auto stream = resources_->getDefaultStreamCurrentDevice();

  if (indicesOptions_ == INDICES_32_BIT) {
    // The data is stored as int32 on the GPU
    FAISS_ASSERT(listId < deviceListIndices_.size());

    auto intInd = deviceListIndices_[listId]->data.copyToHost<int>(stream);

    std::vector<Index::idx_t> out(intInd.size());
    for (size_t i = 0; i < intInd.size(); ++i) {
      out[i] = (Index::idx_t)intInd[i];
    }

    return out;
  } else if (indicesOptions_ == INDICES_64_BIT) {
    // The data is stored as int64 on the GPU
    FAISS_ASSERT(listId < deviceListIndices_.size());

    return deviceListIndices_[listId]->data.copyToHost<Index::idx_t>(stream);
  } else if (indicesOptions_ == INDICES_CPU) {
    // The data is not stored on the GPU
    FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

    auto &userIds = listOffsetToUserIndex_[listId];

    // We should have the same number of indices on the CPU as we do vectors
    // encoded on the GPU
    FAISS_ASSERT(userIds.size() == deviceListData_[listId]->numVecs);

    // this will return a copy
    return userIds;
  } else {
    // unhandled indices type (includes INDICES_IVF)
    FAISS_ASSERT(false);
    return std::vector<Index::idx_t>();
  }
}

std::vector<uint8_t> IMIBase::getListVectorData(int listId,
                                                bool gpuFormat) const {
  FAISS_THROW_IF_NOT_FMT(listId < numLists_,
                         "IVF list %d is out of bounds (%d lists total)",
                         listId, numLists_);
  FAISS_ASSERT(listId < deviceListData_.size());
  FAISS_ASSERT(listId < deviceListLengths_.size());

  auto stream = resources_->getDefaultStreamCurrentDevice();

  auto &list = deviceListData_[listId];
  auto gpuCodes = list->data.copyToHost<uint8_t>(stream);

  if (gpuFormat) {
    return gpuCodes;
  } else {
    // The GPU layout may be different than the CPU layout (e.g., vectors rather
    // than dimensions interleaved), translate back if necessary
    return translateCodesFromGpu_(std::move(gpuCodes), list->numVecs);
  }
}

void IMIBase::addEncodedVectorsToList_(int listId, const void *codes,
                                       const Index::idx_t *indices,
                                       size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // This list must already exist
  FAISS_ASSERT(listId < deviceListData_.size());

  // This list must currently be empty
  auto &listCodes = deviceListData_[listId];
  FAISS_ASSERT(listCodes->data.size() == 0);
  FAISS_ASSERT(listCodes->numVecs == 0);

  // If there's nothing to add, then there's nothing we have to do
  if (numVecs == 0) {
    return;
  }

  // The GPU might have a different layout of the memory
  auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
  auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

  // We only have int32 length representaz3tions on the GPU per each
  // list; the length is in sizeof(char)
  FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());

  // Translate the codes as needed to our preferred form
  std::vector<uint8_t> codesV(cpuListSizeInBytes);
  std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
  auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);

  listCodes->data.append(translatedCodes.data(), gpuListSizeInBytes, stream,
                         true /* exact reserved size */);
  listCodes->numVecs = numVecs;

  // Handle the indices as well
  addIndicesFromCpu_(listId, indices, numVecs);

  deviceListDataPointers_.data()[listId] = listCodes->data.data();
  deviceListLengths_.data()[listId] = numVecs;

  // We update this as well, since the multi-pass algorithm uses it
  maxListLength_ = std::max(maxListLength_, (int)numVecs);

  // device_vector add is potentially happening on a different stream
  // than our default stream
  if (resources_->getDefaultStreamCurrentDevice() != 0) {
    streamWait({stream}, {0});
  }
}

void IMIBase::addIndicesFromCpu_(int listId, const Index::idx_t *indices,
                                 size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // This list must currently be empty
  auto &listIndices = deviceListIndices_[listId];
  FAISS_ASSERT(listIndices->data.size() == 0);
  FAISS_ASSERT(listIndices->numVecs == 0);

  if (indicesOptions_ == INDICES_32_BIT) {
    // Make sure that all indices are in bounds
    std::vector<int> indices32(numVecs);
    for (size_t i = 0; i < numVecs; ++i) {
      auto ind = indices[i];
      FAISS_ASSERT(ind <= (Index::idx_t)std::numeric_limits<int>::max());
      indices32[i] = (int)ind;
    }

    static_assert(sizeof(int) == 4, "");

    listIndices->data.append((uint8_t *)indices32.data(), numVecs * sizeof(int),
                             stream, true /* exact reserved size */);

  } else if (indicesOptions_ == INDICES_64_BIT) {
    listIndices->data.append((uint8_t *)indices, numVecs * sizeof(Index::idx_t),
                             stream, true /* exact reserved size */);
  } else if (indicesOptions_ == INDICES_CPU) {
    // indices are stored on the CPU
    FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

    auto &userIndices = listOffsetToUserIndex_[listId];
    userIndices.insert(userIndices.begin(), indices, indices + numVecs);
  } else {
    // indices are not stored
    FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
  }

  if (indicesOptions_ == INDICES_64_BIT) {
    deviceListIndexPointersIdxT_.data()[listId] =
        (Index::idx_t *)listIndices->data.data();
  } else {
    deviceListIndexPointers_.data()[listId] = (int *)listIndices->data.data();
  }
}

int IMIBase::addVectors(Tensor<float, 2, true> &vecs,
                        Tensor<Index::idx_t, 1, true> &indices) {
  FAISS_ASSERT(vecs.getSize(0) % quantizer_->getNumCodebooks() == 0);
  FAISS_ASSERT(vecs.getSize(1) * quantizer_->getNumCodebooks() == dim_);

  auto stream = resources_->getDefaultStreamCurrentDevice();

  int vecsPerCodebook = vecs.getSize(0) / quantizer_->getNumCodebooks();

  FAISS_ASSERT(vecsPerCodebook == indices.getSize(0));

  // Determine which IVF lists we need to append to

  // We don't actually need this
  DeviceTensor<float, 2, true> listDistance(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {vecsPerCodebook, 1});
  // We use this
  DeviceTensor<ushort2, 2, true> listIds2d(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {vecsPerCodebook, 1});

  quantizer_->query(vecs, 1, listDistance, listIds2d, false);

  // Copy the lists that we wish to append to back to the CPU
  // FIXME: really this can be into pinned memory and a true async
  // copy on a different stream; we can start the copy early, but it's
  // tiny
  auto listIdsHost = listIds2d.copyToVector(stream);

  // Now we add the encoded vectors to the individual lists
  // First, make sure that there is space available for adding the new
  // encoded vectors and indices

  // list id -> vectors being added
  std::unordered_map<int, std::vector<int>> listToVectorIds;

  // vector id -> which list it is being appended to
  std::vector<int> vectorIdToList(vecsPerCodebook);

  // vector id -> offset in list
  // (we already have vector id -> list id in listIds)
  std::vector<int> listOffsetHost(listIdsHost.size());

  // Number of valid vectors that we actually add; we return this
  int numAdded = 0;

  for (int i = 0; i < listIdsHost.size(); ++i) {
    // Add vector could be invalid (contains NaNs etc)
    if (listIdsHost[i].x == std::numeric_limits<unsigned short>::max() ||
        listIdsHost[i].y == std::numeric_limits<unsigned short>::max()) {
      listOffsetHost[i] = -1;
      vectorIdToList[i] = -1;
      continue;
    }

    auto listId = quantizer_->toMultiIndex(listIdsHost[i]);

    FAISS_ASSERT(listId < numLists_);
    ++numAdded;
    vectorIdToList[i] = listId;

    int offset = deviceListData_[listId]->numVecs;

    auto it = listToVectorIds.find(listId);
    if (it != listToVectorIds.end()) {
      offset += it->second.size();
      it->second.push_back(i);
    } else {
      listToVectorIds[listId] = std::vector<int>{i};
    }

    listOffsetHost[i] = offset;
  }

  // If we didn't add anything (all invalid vectors that didn't map to IVF
  // clusters), no need to continue
  if (numAdded == 0) {
    return 0;
  }

  // unique lists being added to
  std::vector<int> uniqueLists;

  for (auto &vecs : listToVectorIds) {
    uniqueLists.push_back(vecs.first);
  }

  std::sort(uniqueLists.begin(), uniqueLists.end());

  // In the same order as uniqueLists, list the vectors being added to that list
  // contiguously
  // (unique list 0 vectors ...)(unique list 1 vectors ...) ...
  std::vector<int> vectorsByUniqueList;

  // For each of the unique lists, the start offset in vectorsByUniqueList
  std::vector<int> uniqueListVectorStart;

  // For each of the unique lists, where we start appending in that list by the
  // vector offset
  std::vector<int> uniqueListStartOffset;

  // For each of the unique lists, find the vectors which should be appended to
  // that list
  for (auto ul : uniqueLists) {
    uniqueListVectorStart.push_back(vectorsByUniqueList.size());

    FAISS_ASSERT(listToVectorIds.count(ul) != 0);

    // The vectors we are adding to this list
    auto &vecs = listToVectorIds[ul];
    vectorsByUniqueList.insert(vectorsByUniqueList.end(), vecs.begin(),
                               vecs.end());

    // How many vectors we previously had (which is where we start appending on
    // the device)
    uniqueListStartOffset.push_back(deviceListData_[ul]->numVecs);
  }

  // We terminate uniqueListVectorStart with the overall number of vectors being
  // added, which could be different than vecs.getSize(0) as some vectors could
  // be invalid
  uniqueListVectorStart.push_back(vectorsByUniqueList.size());

  // We need to resize the data structures for the inverted lists on
  // the GPUs, which means that they might need reallocation, which
  // means that their base address may change. Figure out the new base
  // addresses, and update those in a batch on the device
  {
    // Resize all of the lists that we are appending to
    for (auto &counts : listToVectorIds) {
      auto listId = counts.first;
      int numVecsToAdd = counts.second.size();

      auto &codes = deviceListData_[listId];
      int oldNumVecs = codes->numVecs;
      int newNumVecs = codes->numVecs + numVecsToAdd;

      auto newSizeBytes = getGpuVectorsEncodingSize_(newNumVecs);
      codes->data.resize(newSizeBytes, stream);
      codes->numVecs = newNumVecs;

      auto &indices = deviceListIndices_[listId];
      if ((indicesOptions_ == INDICES_32_BIT) ||
          (indicesOptions_ == INDICES_64_BIT)) {
        size_t indexSize = (indicesOptions_ == INDICES_32_BIT)
                               ? sizeof(int)
                               : sizeof(Index::idx_t);

        indices->data.resize(indices->data.size() + numVecsToAdd * indexSize,
                             stream);
        FAISS_ASSERT(indices->numVecs == oldNumVecs);
        indices->numVecs = newNumVecs;

      } else if (indicesOptions_ == INDICES_CPU) {
        // indices are stored on the CPU side
        FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

        auto &userIndices = listOffsetToUserIndex_[listId];
        userIndices.resize(newNumVecs);
      } else {
        // indices are not stored on the GPU or CPU side
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
      }

      // This is used by the multi-pass query to decide how much scratch
      // space to allocate for intermediate results
      maxListLength_ = std::max(maxListLength_, newNumVecs);
    }

    // Update all pointers and sizes on the device for lists that we
    // appended to
    updateDeviceListInfo_(uniqueLists, stream);
  }

  // If we're maintaining the indices on the CPU side, update our
  // map. We already resized our map above.
  if (indicesOptions_ == INDICES_CPU) {
    // We need to maintain the indices on the CPU side
    HostTensor<Index::idx_t, 1, true> hostIndices(indices, stream);

    for (int i = 0; i < hostIndices.getSize(0); ++i) {
      // Add vector could be invalid (contains NaNs etc)
      if (listIdsHost[i].x == std::numeric_limits<unsigned short>::max() ||
          listIdsHost[i].y == std::numeric_limits<unsigned short>::max()) {
        continue;
      }

      auto listId = quantizer_->toMultiIndex(listIdsHost[i]);

      int offset = listOffsetHost[i];
      FAISS_ASSERT(offset >= 0);

      FAISS_ASSERT(listId < listOffsetToUserIndex_.size());
      auto &userIndices = listOffsetToUserIndex_[listId];

      FAISS_ASSERT(offset < userIndices.size());
      userIndices[offset] = hostIndices[i];
    }
  }

  // Copy the offsets to the GPU
  auto listIdsDevice = listIds2d.downcastOuter<1>();
  auto listOffsetDevice = toDeviceTemporary(resources_, listOffsetHost, stream);
  auto uniqueListsDevice = toDeviceTemporary(resources_, uniqueLists, stream);
  auto vectorsByUniqueListDevice =
      toDeviceTemporary(resources_, vectorsByUniqueList, stream);
  auto uniqueListVectorStartDevice =
      toDeviceTemporary(resources_, uniqueListVectorStart, stream);
  auto uniqueListStartOffsetDevice =
      toDeviceTemporary(resources_, uniqueListStartOffset, stream);

  // Actually encode and append the vectors
  appendVectors_(vecs, indices, uniqueListsDevice, vectorsByUniqueListDevice,
                 uniqueListVectorStartDevice, uniqueListStartOffsetDevice,
                 listIdsDevice, listOffsetDevice, stream);

  // We added this number
  return numAdded;
}

} // namespace gpu
} // namespace faiss

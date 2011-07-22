// Per CPU/Thread data support -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include "Threads.h"
#include "CacheLineStorage.h"

#include <boost/utility.hpp>
#include <cassert>

namespace GaloisRuntime {

//Stores 1 item per thread
//The master thread is thread 0
//Durring Parallel regions the threads index
//from 0 -> num - 1 (one thread pool thread shares an index with the user thread)
template<typename T>
class PerCPU : private boost::noncopyable {
protected:
  cache_line_storage<T>* datum;
  unsigned int num;

  int myID() const {
    int i = ThreadPool::getMyID();
    return std::max(0, i - 1);
  }

public:
  PerCPU()
  {
    num = getSystemThreadPolicy().getNumThreads();
    datum = new cache_line_storage<T>[num];
  }
  explicit PerCPU(const T& ival)
  {
    num = getSystemThreadPolicy().getNumThreads();
    datum = new cache_line_storage<T>[num];
    reset(ival);
  }
  
  virtual ~PerCPU() {
    delete[] datum;
  }

  void reset(const T& d) {
    for (unsigned int i = 0; i < num; ++i)
      datum[i].data = d;
  }

  unsigned int myEffectiveID() const {
    return myID();
  }

  T& get(unsigned int i) {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  const T& get(unsigned int i) const {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  T& get() {
    return get(myID());
  }

  const T& get() const {
    return get(myID());
  }

  T& getNext() {
    return get((myID() + 1) % getSystemThreadPool().getActiveThreads());
  }

  const T& getNext() const {
    return get((myID() + 1) % getSystemThreadPool().getActiveThreads());
  }

  unsigned int size() const {
    return num;
  }
};

template<typename T>
class PerLevel {
  cache_line_storage<T>* datum;
  unsigned int num;
  unsigned int level;
  ThreadPolicy& P;

protected:

  unsigned int myID() const {
    int i = ThreadPool::getMyID();
    return std::max(0, i - 1);
  }


public:
  PerLevel() :P(getSystemThreadPolicy())
  {
    //last iteresting level (should be package)
    level = P.getNumLevels() - 1;
    num = P.getLevelSize(level);
    datum = new cache_line_storage<T>[num];
  }
  
  virtual ~PerLevel() {
    delete[] datum;
  }

  unsigned int myEffectiveID() const {
    return P.indexLevelMap(level, myID());
  }

  T& get(unsigned int i) {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  const T& get(unsigned int i) const {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  T& get() {
    return get(myEffectiveID());
  }

  const T& get() const {
    return get(myEffectiveID());
  }

  unsigned int size() const {
    return num;
  }

  bool isFirstInLevel() const {
    return P.isFirstInLevel(level, myID());
  }

};

}

#endif


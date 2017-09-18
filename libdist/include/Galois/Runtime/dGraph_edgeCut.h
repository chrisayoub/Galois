/** partitioned graph wrapper for edgeCut -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Contains the edge cut functionality to be used in dGraph.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#ifndef _GALOIS_DIST_HGRAPHEC_H
#define _GALOIS_DIST_HGRAPHEC_H

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "Galois/Runtime/dGraph.h"
#include "Galois/Runtime/OfflineGraph.h"
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"

//template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
//class hGraph;

template<typename NodeTy, typename EdgeTy, bool isBipartite = false, 
         bool BSPNode = false, bool BSPEdge = false>
class hGraph_edgeCut : public hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> {
  public:
    typedef hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> base_hGraph;
    // GID = ghostMap[LID - numOwned]
    std::vector<uint64_t> ghostMap;
    // LID = GlobalToLocalGhostMap[GID]
    std::unordered_map<uint64_t, uint32_t> GlobalToLocalGhostMap;
    //LID Node owned by host i. Stores ghost nodes from each host.
    std::vector<std::pair<uint32_t, uint32_t>> hostNodes;

    std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;
    uint32_t numOwned_withoutEdges;
    uint64_t globalOffset_bipartite;

    uint64_t globalOffset;
    uint32_t numNodes;
    uint32_t numOwned_withEdges;

    // Return the local offsets for the nodes to host.
    std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return hostNodes[host];
    }

    // Return the gid offsets assigned to the hosts.
    std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return base_hGraph::gid2host[host];
    }
    std::pair<uint64_t, uint64_t> nodes_by_host_bipartite_G(uint32_t host) const {
          return gid2host_withoutEdges[host];
    }


    // Return the ID to which gid belongs after partition.
    unsigned getHostID(uint64_t gid) const {
      for (auto i = 0U; i < hostNodes.size(); ++i) {
        uint64_t start, end;
        std::tie(start, end) = nodes_by_host_G(i);
        if (gid >= start && gid < end) {
          return i;
        }
        if(isBipartite){
          if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
        return i;
        }
      }
      return -1;
    }

    // Return if gid is Owned by local host.
    bool isOwned(uint64_t gid) const {
      return gid >= globalOffset && gid < globalOffset + numOwned_withEdges;
      if (isBipartite) {
        if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
          return true;
      }
    }

    // Return is gid is present locally (owned or mirror).
    bool isLocal(uint64_t gid) const {
      if (isOwned(gid)) return true;
      return (GlobalToLocalGhostMap.find(gid) != GlobalToLocalGhostMap.end());
    }

    /**
     * Constructor for hGraph_edgeCut
     */
    hGraph_edgeCut(const std::string& filename, 
                   const std::string& partitionFolder, 
                   unsigned host, 
                   unsigned _numHosts, 
                   std::vector<unsigned> scalefactor, 
                   bool transpose = false) : 
                    base_hGraph(host, _numHosts) {
      /*, uint32_t& _numNodes, uint32_t& _numOwned,uint64_t& _numEdges, 
       *  uint64_t& _totalNodes, unsigned _id )*/
      Galois::StatTimer StatTimer_graph_construct("TIME_GRAPH_CONSTRUCT");
      StatTimer_graph_construct.start();
      Galois::StatTimer StatTimer_graph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM");
      uint32_t _numNodes;
      uint64_t _numEdges;

      // only used to determine node splits among hosts; abandonded later
      // for the FileGraph which mmaps appropriate regions of memory
      Galois::Graph::OfflineGraph g(filename);

      base_hGraph::totalNodes = g.size();
      base_hGraph::totalEdges = g.sizeEdges();
      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << 
                   base_hGraph::totalNodes << "\n";

      uint64_t numNodes_to_divide = base_hGraph::computeMasters(g, scalefactor, isBipartite);

      // at this point gid2Host has pairs for how to split nodes among
      // hosts; pair has begin and end
      uint64_t nodeBegin = base_hGraph::gid2host[base_hGraph::id].first;
      typename Galois::Graph::OfflineGraph::edge_iterator edgeBegin = 
        g.edge_begin(nodeBegin);

      uint64_t nodeEnd = base_hGraph::gid2host[base_hGraph::id].second;
      typename Galois::Graph::OfflineGraph::edge_iterator edgeEnd = 
        g.edge_begin(nodeEnd);

      numOwned_withEdges = base_hGraph::totalOwnedNodes = 
                           base_hGraph::numOwned = 
                           (nodeEnd - nodeBegin);
      
      // file graph that is mmapped for much faster reading; will use this
      // when possible from now on in the code
      Galois::Graph::FileGraph fileGraph;

      fileGraph.partFromFile(filename,
        std::make_pair(boost::make_counting_iterator<uint64_t>(nodeBegin), 
                       boost::make_counting_iterator<uint64_t>(nodeEnd)),
        std::make_pair(edgeBegin, edgeEnd), true);

      // TODO
      // currently not being used, may not be updated
      if (isBipartite) {
        uint64_t numNodes_without_edges = (g.size() - numNodes_to_divide);
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i) {
          auto p = Galois::block_range(
                     0U, (unsigned)numNodes_without_edges, i, 
                     base_hGraph::numHosts);

          std::cout << " last node : " << base_hGraph::last_nodeID_withEdges_bipartite << 
                       ", " << p.first << " , " << p.second << "\n";

          gid2host_withoutEdges.push_back(std::make_pair(base_hGraph::last_nodeID_withEdges_bipartite + p.first + 1, base_hGraph::last_nodeID_withEdges_bipartite + p.second + 1));
          globalOffset_bipartite = gid2host_withoutEdges[base_hGraph::id].first;
        }

        numOwned_withoutEdges = (gid2host_withoutEdges[base_hGraph::id].second - 
                                 gid2host_withoutEdges[base_hGraph::id].first);
        base_hGraph::totalOwnedNodes = base_hGraph::numOwned = 
                                   (nodeEnd - nodeBegin) + 
                                   (gid2host_withoutEdges[base_hGraph::id].second - 
                                    gid2host_withoutEdges[base_hGraph::id].first);
      }

      globalOffset = nodeBegin;

      std::cerr << "[" << base_hGraph::id << "] Owned nodes: " << 
                   base_hGraph::numOwned << "\n";

      _numEdges = edgeEnd - edgeBegin;
      std::cerr << "[" << base_hGraph::id << "] Total edges : " << 
                          _numEdges << "\n";

      Galois::DynamicBitSet ghosts;
      ghosts.resize(g.size());

      auto activeThreads = Galois::Runtime::activeThreads;
      Galois::setActiveThreads(numFileThreads); // only use limited threads for reading file

      Galois::Timer timer;
      timer.start();
      fileGraph.reset_byte_counters();

      // vector to hold a prefix sum for use in thread work distribution
      std::vector<uint64_t> prefixSumOfEdges(base_hGraph::numOwned);

      // loop through all nodes we own and determine ghosts (note a node
      // we own can also be marked a ghost here if there's an outgoing edge to 
      // it)
      // Also determine prefix sums
      auto edgeOffset = fileGraph.edge_begin(nodeBegin);
      auto beginIter = boost::make_counting_iterator(nodeBegin);
      auto endIter = boost::make_counting_iterator(nodeEnd);
      Galois::do_all(
        beginIter, endIter,
        [&] (auto n) {
          auto ii = fileGraph.edge_begin(n);
          auto ee = fileGraph.edge_end(n);
          for (; ii < ee; ++ii) {
            ghosts.set(fileGraph.getEdgeDst(ii));
          }
          prefixSumOfEdges[n - nodeBegin] = std::distance(edgeOffset, ee);
        },
        Galois::loopname("EdgeInspection"),
        Galois::timeit(),
        Galois::do_all_steal<true>(),
        Galois::no_stats()
      );

      timer.stop();
      fprintf(stderr, "[%u] Edge inspection time : %f seconds to read %lu bytes (%f MBPS)\n", 
          base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph.num_bytes_read(), fileGraph.num_bytes_read()/(float)timer.get_usec());

      Galois::setActiveThreads(activeThreads); // revert to prior active threads

      // only nodes we do not own are actual ghosts (i.e. filter the "ghosts"
      // found above)
      for (uint64_t x = 0; x < g.size(); ++x)
        if (ghosts.test(x) && !isOwned(x))
          ghostMap.push_back(x);
      std::cerr << "[" << base_hGraph::id << "] Ghost nodes: " << 
                   ghostMap.size() << "\n";

      hostNodes.resize(base_hGraph::numHosts, std::make_pair(~0, ~0));

      // determine on which hosts each ghost nodes resides
      GlobalToLocalGhostMap.reserve(ghostMap.size());
      for (unsigned ln = 0; ln < ghostMap.size(); ++ln) {
        unsigned lid = ln + base_hGraph::numOwned;
        auto gid = ghostMap[ln];
        GlobalToLocalGhostMap[gid] = lid;

        for (auto h = 0U; h < base_hGraph::gid2host.size(); ++h) {
          auto& p = base_hGraph::gid2host[h];
          if (gid >= p.first && gid < p.second) {
            hostNodes[h].first = std::min(hostNodes[h].first, lid);
            hostNodes[h].second = lid + 1;
            break;
          } else if (isBipartite) {
            auto& p2 = gid2host_withoutEdges[h];
            if(gid >= p2.first && gid < p2.second) {
              hostNodes[h].first = std::min(hostNodes[h].first, lid);
              hostNodes[h].second = lid + 1;
              break;
             }
          }
        }
      }

      base_hGraph::numNodes = numNodes = 
                              _numNodes = 
                              base_hGraph::numOwned + ghostMap.size();

      assert((uint64_t)base_hGraph::numOwned + (uint64_t)ghostMap.size() == 
             (uint64_t)numNodes);
      prefixSumOfEdges.resize(_numNodes, prefixSumOfEdges.back());

      // transpose is usually used for incoming edge cuts: this makes it
      // so you consider ghosts as having edges as well (since in IEC ghosts
      // have outgoing edges)
      if (transpose) {
        base_hGraph::numNodesWithEdges = base_hGraph::numNodes;
      } else {
        base_hGraph::numNodesWithEdges = base_hGraph::numOwned;
      }

      base_hGraph::beginMaster = 0;
      base_hGraph::endMaster = base_hGraph::numOwned;

      //std::cerr << "[" << base_hGraph::id << "] Beginning memory allocation" <<
      //             "\n";

      if (!edgeNuma) {
        base_hGraph::graph.allocateFrom(_numNodes, _numEdges);
      } else {
        // determine division of nodes among threads and allocate based on that
        printf("Edge based NUMA division on\n");
        //base_hGraph::graph.allocateFrom(_numNodes, _numEdges, prefixSumOfEdges);
        base_hGraph::graph.allocateFromByNode(_numNodes, _numEdges, 
                                              prefixSumOfEdges);
      }
      //std::cerr << "[" << base_hGraph::id << "] Allocate done" << "\n";

      base_hGraph::graph.constructNodes();
      //std::cerr << "[" << base_hGraph::id << "] Construct nodes done" << "\n";

      auto beginIter2 = boost::make_counting_iterator((uint32_t)0);
      auto endIter2 = boost::make_counting_iterator(numNodes);
      auto& base_graph = base_hGraph::graph;
      Galois::do_all(
        beginIter2, endIter2,
        [&] (auto n) {
          base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
        },
        Galois::loopname("EdgeLoading"),
        Galois::do_all_steal<true>(),
        Galois::timeit(),
        Galois::no_stats()
      );


      loadEdges(base_hGraph::graph, fileGraph);
      std::cerr << "[" << base_hGraph::id << "] Edges loaded" << "\n";
      
      if (transpose) {
        base_hGraph::graph.transpose(edgeNuma);
        base_hGraph::transposed = true;
      }

      fill_mirrorNodes(base_hGraph::mirrorNodes);

      // !transpose because tranpose finds thread ranges for you
      if (!edgeNuma && !transpose) {
        Galois::StatTimer StatTimer_thread_ranges("TIME_THREAD_RANGES");

        StatTimer_thread_ranges.start();

        base_hGraph::determine_thread_ranges(_numNodes, prefixSumOfEdges);

        // experimental test of new thread ranges
        //base_hGraph::determine_thread_ranges(0, _numNodes, 
        //                              base_hGraph::graph.getThreadRangesVector());

        StatTimer_thread_ranges.stop();
      }

      // find ranges for master + nodes with edges
      base_hGraph::determine_thread_ranges_master();
      base_hGraph::determine_thread_ranges_with_edges();
      base_hGraph::initialize_specific_ranges();

      StatTimer_graph_construct.stop();

      StatTimer_graph_construct_comm.start();
      base_hGraph::setup_communication();
      StatTimer_graph_construct_comm.stop();
    }

  uint32_t G2L(uint64_t gid) const {
    if (gid >= globalOffset && gid < globalOffset + numOwned_withEdges)
      return gid - globalOffset;

    if(isBipartite){
      if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
            return gid - globalOffset_bipartite + numOwned_withEdges;
    }

    return GlobalToLocalGhostMap.at(gid);
#if 0
    auto ii = std::lower_bound(ghostMap.begin(), ghostMap.end(), gid);
    assert(*ii == gid);
    return std::distance(ghostMap.begin(), ii) + base_hGraph::numOwned;
#endif
  }

  uint64_t L2G(uint32_t lid) const {
    assert(lid < numNodes);
    if (lid < numOwned_withEdges)
      return lid + globalOffset;
    if(isBipartite){
      if(lid >= numOwned_withEdges && lid < base_hGraph::numOwned)
        return lid + globalOffset_bipartite;
    }
    return ghostMap[lid - base_hGraph::numOwned];
  }

  template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdges(GraphTy& graph, Galois::Graph::FileGraph& fileGraph) {
    if (base_hGraph::id == 0) {
      fprintf(stderr, "Loading edge-data while creating edges.\n");
    }

    Galois::Timer timer;
    timer.start();
    fileGraph.reset_byte_counters();

    auto beginIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].first);
    auto endIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].second);
    Galois::do_all(
      beginIter, endIter,
      [&] (auto n) {
        auto ii = fileGraph.edge_begin(n);
        auto ee = fileGraph.edge_end(n);
        uint32_t lsrc = this->G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
        for (; ii < ee; ++ii) {
          auto gdst = fileGraph.getEdgeDst(ii);
          decltype(gdst) ldst = this->G2L(gdst);
          auto gdata = fileGraph.getEdgeData<typename GraphTy::edge_data_type>(ii);
          graph.constructEdge(cur++, ldst, gdata);
        }
        assert(cur == (*graph.edge_end(lsrc)));
      },
      Galois::loopname("EdgeLoading"),
      Galois::do_all_steal<true>(),
      Galois::timeit(),
      Galois::no_stats()
    );

    timer.stop();
    fprintf(stderr, "[%u] Edge loading time : %f seconds to read %lu bytes (%f MBPS)\n", 
        base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph.num_bytes_read(), fileGraph.num_bytes_read()/(float)timer.get_usec());
  }

  template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdges(GraphTy& graph, Galois::Graph::FileGraph& fileGraph) {
    if (base_hGraph::id == 0) {
      fprintf(stderr, "Loading void edge-data while creating edges.\n");
    }

    Galois::Timer timer;
    timer.start();
    fileGraph.reset_byte_counters();

    auto beginIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].first);
    auto endIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].second);
    Galois::do_all(
      beginIter, endIter,
      [&] (auto n) {
        auto ii = fileGraph.edge_begin(n);
        auto ee = fileGraph.edge_end(n);
        uint32_t lsrc = this->G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
        for (; ii < ee; ++ii) {
          auto gdst = fileGraph.getEdgeDst(ii);
          decltype(gdst) ldst = this->G2L(gdst);
          graph.constructEdge(cur++, ldst);
        }
        assert(cur == (*graph.edge_end(lsrc)));
      },
      Galois::loopname("EdgeLoading"),
      Galois::do_all_steal<true>(),
      Galois::timeit(),
      Galois::no_stats()
    );


    timer.stop();
    fprintf(stderr, "[%u] Edge loading time : %f seconds to read %lu bytes (%f MBPS)\n", 
        base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph.num_bytes_read(), fileGraph.num_bytes_read()/(float)timer.get_usec());
  }

  void fill_mirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes){
    for(uint32_t h = 0; h < hostNodes.size(); ++h){
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(h);
      for(; start != end; ++start){
        mirrorNodes[h].push_back(L2G(start));
      }
    }
  }

  std::string getPartitionFileName(const std::string& filename, const std::string & basename, unsigned hostID, unsigned num_hosts){
    return filename;
  }

  bool is_vertex_cut() const{
    return false;
  }

  /*
   * Returns the total nodes : master + slaves created on the local host.
   */
  uint64_t get_local_total_nodes() const {
    return (base_hGraph::numOwned + base_hGraph::totalMirrorNodes);
  }

  void reset_bitset(typename base_hGraph::SyncType syncType, void (*bitset_reset_range)(size_t, size_t)) const {
    if (syncType == base_hGraph::syncBroadcast) { // reset masters
      if (numOwned_withEdges > 0) {
        bitset_reset_range(0, numOwned_withEdges - 1);
      }
    } else { // reset mirrors
      assert(syncType == base_hGraph::syncReduce);
      if (numOwned_withEdges < numNodes) {
        bitset_reset_range(numOwned_withEdges, numNodes - 1);
      }
    }
  }
};
#endif

# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import enum
import time
from typing import Dict, List, Optional, Tuple

from fastapi import Request
from uhashring import HashRing

from vllm_router.log import init_logger
from vllm_router.service_discovery import EndpointInfo
from vllm_router.stats.engine_stats import EngineStats
from vllm_router.stats.request_stats import RequestStats
from vllm_router.utils import SingletonABCMeta
from collections import OrderedDict

logger = init_logger(__name__)


class RoutingLogic(str, enum.Enum):
    ROUND_ROBIN = "roundrobin"
    SESSION_BASED = "session"
    CACHE_AWARE_LOAD_BALANCING = "cache_aware_load_balancing"


class RoutingInterface(metaclass=SingletonABCMeta):
    @abc.abstractmethod
    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        raise NotImplementedError


class RoundRobinRouter(RoutingInterface):
    # TODO (ApostaC): when available engines in the endpoints changes, the
    # algorithm may not be "perfectly" round-robin.
    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self.req_id = 0
        self._initialized = True

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL using a simple
        round-robin algorithm

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        len_engines = len(endpoints)
        chosen = sorted(endpoints, key=lambda e: e.url)[self.req_id % len_engines]
        self.req_id += 1
        return chosen.url


class SessionRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL based on the session key
    in the request headers
    """

    def __init__(self, session_key: str = None):
        if hasattr(self, "_initialized"):
            return
        if session_key is None:
            raise ValueError("SessionRouter must be initialized with a session_key")
        self.session_key = session_key
        self.hash_ring = HashRing()
        self._initialized = True

    def _qps_routing(
        self, endpoints: List[EndpointInfo], request_stats: Dict[str, RequestStats]
    ) -> str:
        """
        Route the request to the appropriate engine URL based on the QPS of
        each engine

        Args:
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
        """
        lowest_qps = float("inf")
        ret = None
        for info in endpoints:
            url = info.url
            if url not in request_stats:
                return url  # This engine does not have any requests
            request_stat = request_stats[url]
            if request_stat.qps < lowest_qps:
                lowest_qps = request_stat.qps
                ret = url
        return ret

    def _update_hash_ring(self, endpoints: List["EndpointInfo"]):
        """
        Update the hash ring with the current list of endpoints.
        """
        # Extract endpoint URLs
        endpoint_urls = [endpoint.url for endpoint in endpoints]

        # Get the current nodes in the hash ring
        current_nodes = set(self.hash_ring.get_nodes())

        # Convert the new endpoint URLs to a set for easy comparison
        new_nodes = set(endpoint_urls)

        # Remove nodes that are no longer in the list
        for node in current_nodes - new_nodes:
            self.hash_ring.remove_node(node)

        # Add new nodes that are not already in the hash ring
        for node in new_nodes - current_nodes:
            self.hash_ring.add_node(node)

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL by the 'session id' in
        the request headers.
        If there is no session id in the request header, it will pick a server
        with lowest qps

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        session_id = request.headers.get(self.session_key, None)
        logger.debug(f"Got session id: {session_id}")

        # Update the hash ring with the current list of endpoints
        self._update_hash_ring(endpoints)

        if session_id is None:
            # Route based on QPS if no session ID is present
            url = self._qps_routing(endpoints, request_stats)
        else:
            # Use the hash ring to get the endpoint for the session ID
            url = self.hash_ring.get_node(session_id)

        return url
    

class LRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # 标记为最近使用
        return self.cache[key]

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # 移除最旧的条目


class CacheAwareLoadBalancingRouter(RoutingInterface):
    """
    结合负载均衡和KV Cache命中率感知的路由算法
    
    该算法考虑三个关键因素：
    1. 引擎负载（排队请求数、运行请求数）
    2. 预估的KV缓存命中率（针对特定会话）
    """
    
    # KV缓存配置常量
    DEFAULT_BLOCK_REUSE_TIMEOUT = 60  # KV缓存块复用超时（秒）
    
    def __init__(self, session_key: str = None, block_reuse_timeout: int = None):
        if hasattr(self, "_initialized"):
            return
            
        if session_key is None:
            raise ValueError("CacheAwareLoadBalancingRouter must be initialized with a session_key")
            
        self.session_key = session_key
        self.block_reuse_timeout = block_reuse_timeout or self.DEFAULT_BLOCK_REUSE_TIMEOUT
        
        # 会话状态跟踪，使用LRUCache替代普通字典
        self.session_last_time = LRUCache(max_size=150000)  # 会话最后访问时间 {session_id: timestamp}
        self.session_to_engine = LRUCache(max_size=150000)  # 会话到引擎的映射 {session_id: engine_url}

        self.req_id = 0  # 请求ID，用于轮询选择
        
        self._initialized = True
    
    def _predict_cache_hit_rate(self, session_id: str, engine_url: str, 
                               engine_stats: Dict[str, EngineStats]) -> float:
        """
        预测特定会话在特定引擎上的KV缓存命中率
        
        基于以下因素：
        1. 会话最后访问时间（时间间隔越短，命中率越高）
        2. 上次访问到现在间隔的独立会话数（中间访问的会话越多，命中率越低）(暂时不考虑)
        
        返回值：预估的命中率（0.0到1.0之间）
        """
        current_time = time.time()
        
        # 如果会话首次访问或未找到记录，返回0（无命中）
        last_time = self.session_last_time.get(session_id)
        if last_time is None:
            return 0.0
            
        # 检查会话是否使用过此引擎
        stored_engine = self.session_to_engine.get(session_id)
        if stored_engine != engine_url:
            return 0.0
            
        # 计算距离上次访问的时间间隔
        time_since_last_request = current_time - last_time
        
        # 估计KV缓存命中率
            
        # 如果在block_reuse_timeout内，认为缓存完全命中
        if time_since_last_request < self.block_reuse_timeout:
            hit_rate = 1.0
            logger.debug(f"会话 {session_id} 在引擎 {engine_url} 的命中率预测: {hit_rate:.2f}")
            return hit_rate
        
        # 默认返回0
        return 0.0
    
    def _calculate_engine_load_score(self, engine_url: str, 
                                    engine_stats: Dict[str, EngineStats], 
                                    request_stats: Dict[str, RequestStats]) -> float:
        """
        计算引擎的负载分数
        
        分数越低表示引擎负载越轻

        负载因素：负载得分（正在运行的请求数*0.02 + 排队请求数*0.1）
        """
        if engine_url not in engine_stats:
            return 0.0  # 无统计数据，假设负载为0
            
        # 获取引擎统计数据
        stats = engine_stats[engine_url]
        
        # 基本负载因素：运行请求数和排队请求数
        running_load = stats.num_running_requests * 0.02  # 运行中请求权重
        queuing_load = stats.num_queuing_requests * 0.1  # 排队请求权重（略高）
        
        # 暂时不考虑QPS
        # qps_factor = 0.0
        # if engine_url in request_stats:
        #     qps = request_stats[engine_url].qps
        #     qps_factor = qps * 0.2  # QPS权重较低
        
        # 计算总负载分数
        total_load_score = running_load + queuing_load
        
        return total_load_score
    
    def _update_session_info(self, session_id: str, engine_url: str):
        """
        更新会话信息，并根据时间间隔触发清理
        """
        current_time = time.time()
        
        # 更新会话最后访问时间
        self.session_last_time.set(session_id, current_time)
        
        # 更新会话到引擎的映射
        self.session_to_engine.set(session_id, engine_url)
    
    def _select_best_engine(self, session_id: str, endpoints: List[EndpointInfo],
                          engine_stats: Dict[str, EngineStats],
                          request_stats: Dict[str, RequestStats]) -> str:
        """
        选择最佳引擎，综合考虑负载均衡和缓存命中率
        
        对于每个引擎计算综合分数，选择分数最低的
        """
        best_engine_url = None
        best_score = float('inf')
        
        # 缓存命中权重因子
        cache_weight = 0.5  # 50%的权重给缓存命中
        load_weight = 0.5   # 50%的权重给负载均衡
        
        for info in endpoints:
            engine_url = info.url
            
            # 计算负载分数（分数越低越好）
            load_score = self._calculate_engine_load_score(engine_url, engine_stats, request_stats)
            
            # 预测缓存命中率（越高越好）
            cache_hit_rate = self._predict_cache_hit_rate(session_id, engine_url, engine_stats)
            
            # 将命中率转换为分数（分数越低越好）
            cache_score = 1.0 - cache_hit_rate
            
            # 计算综合得分，加权平均
            combined_score = (cache_score * cache_weight) + (load_score * load_weight)
            
            logger.debug(f"引擎 {engine_url} - 负载: {load_score:.2f}, 预估命中率: {cache_hit_rate:.2f}, 综合分数: {combined_score:.2f}")
            
            # 更新最佳引擎
            if combined_score < best_score:
                best_score = combined_score
                best_engine_url = engine_url
        
        # 如果找不到适合的引擎（极少发生），选择负载最低的
        if best_engine_url is None and endpoints:
            best_engine_url = min(
                [e.url for e in endpoints],
                key=lambda url: self._calculate_engine_load_score(url, engine_stats, request_stats)
            )
            
        return best_engine_url

    def _select_best_engine_new(self, session_id: str, endpoints: List[EndpointInfo],
                          engine_stats: Dict[str, EngineStats],
                          request_stats: Dict[str, RequestStats]) -> str:
        """
        若缓存命中大于1，则选择该引擎
        否则，选择根据roundrobin选择的引擎
        """
        for info in endpoints:
            engine_url = info.url
            
            # 预测缓存命中率（越高越好）
            cache_hit_rate = self._predict_cache_hit_rate(session_id, engine_url, engine_stats)
            
            # 如果命中率为1，直接返回该引擎
            if cache_hit_rate >= 1.0:
                logger.debug(f"会话 {session_id} 缓存命中引擎: {engine_url}")
                return engine_url
            
        # 如果没有命中率为1的引擎，使用roundrobin选择
        len_engines = len(endpoints)
        chosen = sorted(endpoints, key=lambda e: e.url)[self.req_id % len_engines]
        self.req_id += 1
        return chosen.url
        
    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        智能路由请求，结合负载感知和缓存命中率预测
        
        对于有会话ID的请求，会基于KV缓存命中率预测和负载状况进行智能选择
        对于无会话ID的请求，会纯粹基于负载均衡选择引擎
        """
        # 提取会话ID
        session_id = request.headers.get(self.session_key, None)
        logger.debug(f"Got session id: {session_id}")
        
        if session_id is None:
            # 无会话ID，使用纯负载均衡策略
            engine_url = min(
                [e.url for e in endpoints],
                key=lambda url: self._calculate_engine_load_score(url, engine_stats, request_stats)
            )
        else:
            # 有会话ID，使用综合策略
            engine_url = self._select_best_engine_new(session_id, endpoints, engine_stats, request_stats)
            
            # 更新会话信息
            self._update_session_info(session_id, engine_url)
        
        return engine_url


# Instead of managing a global _global_router, we can define the initialization functions as:
def initialize_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    if routing_logic == RoutingLogic.ROUND_ROBIN:
        logger.info("Initializing round-robin routing logic")
        return RoundRobinRouter()
    elif routing_logic == RoutingLogic.SESSION_BASED:
        logger.info(f"Initializing session-based routing logic with kwargs: {kwargs}")
        return SessionRouter(kwargs.get("session_key"))
    elif routing_logic == RoutingLogic.CACHE_AWARE_LOAD_BALANCING:
        logger.info(f"Initializing cache-aware load balancing routing logic with kwargs: {kwargs}")
        router = CacheAwareLoadBalancingRouter(
            kwargs.get("session_key"),
            kwargs.get("block_reuse_timeout")
        )
        return router
    else:
        raise ValueError(f"Invalid routing logic {routing_logic}")


def reconfigure_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    # Remove the existing routers from the singleton registry
    for cls in (SessionRouter, RoundRobinRouter, CacheAwareLoadBalancingRouter):
        if cls in SingletonABCMeta._instances:
            del SingletonABCMeta._instances[cls]
    return initialize_routing_logic(routing_logic, *args, **kwargs)


def get_routing_logic() -> RoutingInterface:
    # Look up in our singleton registry which router (if any) has been created.
    for cls in (SessionRouter, RoundRobinRouter, CacheAwareLoadBalancingRouter):
        if cls in SingletonABCMeta._instances:
            return cls()
    raise ValueError("The global router has not been initialized")

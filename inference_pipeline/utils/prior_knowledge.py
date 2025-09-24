"""
Prior Knowledge Utilities for PyTetrad Algorithms

This module provides utilities for extracting and formatting prior knowledge
from dataset metadata for use with PyTetrad algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
import logging

logger = logging.getLogger(__name__)


class PriorKnowledgeFormatter:
    """Formats prior knowledge from metadata for PyTetrad algorithms."""
    
    def __init__(self, metadata: Dict):
        """
        Initialize with dataset metadata.
        
        Args:
            metadata: Dataset metadata containing temporal_order, station_blocks, 
                     root_nodes, etc.
        """
        self.metadata = metadata
        self.temporal_order = metadata.get('temporal_order', [])
        self.station_blocks = metadata.get('station_blocks', [])
        self.station_names = metadata.get('station_names', [])
        self.station_map = metadata.get('station_map', {})
        self.root_nodes = metadata.get('root_nodes', [])
        self.num_nodes = metadata.get('num_nodes', 0)
        
    def extract_temporal_ordering(self) -> List[Tuple[str, str]]:
        """
        Extract temporal ordering constraints from station-wise ordering.
        
        Returns:
            List of (earlier_node, later_node) tuples representing temporal precedence
        """
        if not self.temporal_order or len(self.temporal_order) < 2:
            return []
        
        temporal_constraints = []
        
        # Create constraints based on temporal order
        for i in range(len(self.temporal_order) - 1):
            for j in range(i + 1, len(self.temporal_order)):
                earlier_node = self.temporal_order[i]
                later_node = self.temporal_order[j]
                temporal_constraints.append((earlier_node, later_node))
        
        logger.debug(f"Extracted {len(temporal_constraints)} temporal ordering constraints")
        return temporal_constraints
    
    def extract_station_constraints(self) -> List[Tuple[str, str]]:
        """
        Extract station-based constraints (nodes in earlier stations cannot be 
        caused by nodes in later stations).
        
        Returns:
            List of (later_station_node, earlier_station_node) tuples for forbidden edges
        """
        if not self.station_blocks or len(self.station_blocks) < 2:
            return []
        
        station_constraints = []
        
        # For each station, nodes in later stations cannot cause nodes in earlier stations
        for i in range(len(self.station_blocks)):
            for j in range(i + 1, len(self.station_blocks)):
                later_station_nodes = self.station_blocks[j]
                earlier_station_nodes = self.station_blocks[i]
                
                for later_node in later_station_nodes:
                    for earlier_node in earlier_station_nodes:
                        station_constraints.append((later_node, earlier_node))
        
        logger.debug(f"Extracted {len(station_constraints)} station-based constraints")
        return station_constraints
    
    def extract_root_node_constraints(self) -> List[str]:
        """
        Extract root node constraints (root nodes have no incoming edges).
        
        Returns:
            List of root node names
        """
        if not self.root_nodes:
            return []
        
        logger.debug(f"Extracted {len(self.root_nodes)} root node constraints")
        return self.root_nodes
    
    def get_forbidden_edges(self) -> List[Tuple[str, str]]:
        """
        Get all forbidden edges based on temporal and station constraints.
        
        Returns:
            List of (source, target) tuples representing forbidden edges
        """
        forbidden_edges = []
        
        # Add station-based constraints (later stations cannot cause earlier stations)
        forbidden_edges.extend(self.extract_station_constraints())
        
        # Note: We don't add temporal constraints as forbidden edges because
        # temporal order doesn't necessarily imply causal order
        
        logger.debug(f"Total forbidden edges: {len(forbidden_edges)}")
        return forbidden_edges
    
    def get_required_edges(self) -> List[Tuple[str, str]]:
        """
        Get required edges (if any). Currently empty but can be extended.
        
        Returns:
            List of (source, target) tuples representing required edges
        """
        # Currently no required edges, but this can be extended
        # based on domain knowledge or other metadata
        return []
    
    def get_tier_ordering(self) -> List[List[str]]:
        """
        Get tier ordering for algorithms that support it.
        
        Returns:
            List of tiers, where each tier is a list of node names
        """
        if not self.station_blocks:
            # Fallback to temporal order if no station blocks
            if self.temporal_order:
                return [[node] for node in self.temporal_order]
            return []
        
        # Use station blocks as tiers
        return [list(tier) for tier in self.station_blocks]
    
    def get_prior_knowledge_dict(self) -> Dict:
        """
        Get complete prior knowledge dictionary for PyTetrad algorithms.
        
        Returns:
            Dictionary containing all prior knowledge constraints
        """
        prior_knowledge = {
            'forbidden_edges': self.get_forbidden_edges(),
            'required_edges': self.get_required_edges(),
            'tier_ordering': self.get_tier_ordering(),
            'root_nodes': self.extract_root_node_constraints(),
            'temporal_order': self.temporal_order,
            'station_blocks': self.station_blocks
        }
        
        logger.debug(f"Generated prior knowledge: {len(prior_knowledge['forbidden_edges'])} forbidden edges, "
                    f"{len(prior_knowledge['required_edges'])} required edges, "
                    f"{len(prior_knowledge['tier_ordering'])} tiers")
        
        return prior_knowledge


def create_tetrad_prior_knowledge(prior_knowledge: Dict, node_names: List[str]) -> Optional[object]:
    """
    Create Tetrad PriorKnowledge object from formatted prior knowledge.
    
    Args:
        prior_knowledge: Prior knowledge dictionary
        node_names: List of node names in the dataset
        
    Returns:
        Tetrad PriorKnowledge object or None if not available
    """
    try:
        import jpype
        import edu.cmu.tetrad.graph as graph
        
        if not jpype.isJVMStarted():
            logger.warning("JVM not started, cannot create Tetrad PriorKnowledge")
            return None
        
        # Create PriorKnowledge object
        prior = graph.PriorKnowledge()
        
        # Add forbidden edges
        for source, target in prior_knowledge.get('forbidden_edges', []):
            if source in node_names and target in node_names:
                prior.setForbidden(source, target)
        
        # Add required edges
        for source, target in prior_knowledge.get('required_edges', []):
            if source in node_names and target in node_names:
                prior.setRequired(source, target)
        
        # Add tier ordering
        tier_ordering = prior_knowledge.get('tier_ordering', [])
        if tier_ordering:
            for i, tier in enumerate(tier_ordering):
                for node in tier:
                    if node in node_names:
                        prior.setTierForbidden(node, i)
        
        logger.debug(f"Created Tetrad PriorKnowledge with {len(prior_knowledge.get('forbidden_edges', []))} "
                    f"forbidden edges and {len(tier_ordering)} tiers")
        
        return prior
        
    except Exception as e:
        logger.warning(f"Could not create Tetrad PriorKnowledge: {e}")
        return None


def format_prior_knowledge_for_algorithm(metadata: Dict, algorithm_name: str) -> Dict:
    """
    Format prior knowledge specifically for a given algorithm.
    
    Args:
        metadata: Dataset metadata
        algorithm_name: Name of the algorithm
        
    Returns:
        Formatted prior knowledge dictionary
    """
    formatter = PriorKnowledgeFormatter(metadata)
    prior_knowledge = formatter.get_prior_knowledge_dict()
    
    # Algorithm-specific formatting
    if algorithm_name.lower() in ['pc', 'fci', 'rfci', 'gfci', 'cfci']:
        # Constraint-based algorithms: focus on forbidden edges and tier ordering
        return {
            'forbidden_edges': prior_knowledge['forbidden_edges'],
            'tier_ordering': prior_knowledge['tier_ordering'],
            'root_nodes': prior_knowledge['root_nodes']
        }
    elif algorithm_name.lower() in ['ges', 'fges']:
        # Score-based algorithms: can use all constraints
        return prior_knowledge
    else:
        # Default: return all available prior knowledge
        return prior_knowledge


def validate_prior_knowledge(prior_knowledge: Dict, node_names: List[str]) -> bool:
    """
    Validate that prior knowledge is consistent with dataset.
    
    Args:
        prior_knowledge: Prior knowledge dictionary
        node_names: List of node names in the dataset
        
    Returns:
        True if valid, False otherwise
    """
    node_set = set(node_names)
    
    # Check forbidden edges
    for source, target in prior_knowledge.get('forbidden_edges', []):
        if source not in node_set or target not in node_set:
            logger.warning(f"Forbidden edge ({source}, {target}) references unknown nodes")
            return False
    
    # Check required edges
    for source, target in prior_knowledge.get('required_edges', []):
        if source not in node_set or target not in node_set:
            logger.warning(f"Required edge ({source}, {target}) references unknown nodes")
            return False
    
    # Check tier ordering
    for tier in prior_knowledge.get('tier_ordering', []):
        for node in tier:
            if node not in node_set:
                logger.warning(f"Tier ordering references unknown node: {node}")
                return False
    
    return True


def log_prior_knowledge_summary(prior_knowledge: Dict, dataset_name: str = "dataset"):
    """Log a summary of the prior knowledge being used."""
    logger.info(f"Prior knowledge for {dataset_name}:")
    logger.info(f"  Forbidden edges: {len(prior_knowledge.get('forbidden_edges', []))}")
    logger.info(f"  Required edges: {len(prior_knowledge.get('required_edges', []))}")
    logger.info(f"  Tier ordering: {len(prior_knowledge.get('tier_ordering', []))} tiers")
    logger.info(f"  Root nodes: {len(prior_knowledge.get('root_nodes', []))}")
    
    if prior_knowledge.get('forbidden_edges'):
        logger.debug(f"  Forbidden edges: {prior_knowledge['forbidden_edges'][:5]}...")
    
    if prior_knowledge.get('tier_ordering'):
        logger.debug(f"  Tier structure: {[len(tier) for tier in prior_knowledge['tier_ordering']]}")

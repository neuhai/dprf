"""
Core module for the Generalized DPRF framework.

This module contains the core components of the Dynamic Persona Refinement Framework:
- DPRFAgent: The main agent class that handles the DPRF process
- PersonaRefiner: Class for refining personas based on analysis
- Utility functions for file handling, prompt formatting, etc.
"""

from .dprf_agent import DPRFAgent
from .persona_refinement import PersonaRefiner
from .utils import (
    load_json,
    save_json,
    ensure_directory,
    format_persona_prompt,
    format_analysis_prompt,
    format_refinement_prompt,
    format_analysis_refinement_prompt,
    format_direct_refinement_prompt,
    format_peer_review_instruction
)

__all__ = [
    'DPRFAgent',
    'PersonaRefiner',
    'load_json',
    'save_json',
    'ensure_directory',
    'format_persona_prompt',
    'format_analysis_prompt',
    'format_refinement_prompt',
    'format_analysis_refinement_prompt',
    'format_direct_refinement_prompt',
    'format_peer_review_instruction'
] 
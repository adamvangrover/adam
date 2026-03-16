from abc import abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from pydantic import BaseModel, ValidationError
import logging

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class PydanticAgentBase(AgentBase):
    """
    Unified base class for System 2 agents requiring strictly typed 
    Pydantic input/output schemas.
    
    Subclasses must implement the `execute_pydantic` method.
    """
    
    @abstractmethod
    async def execute_pydantic(self, input_data: AgentInput) -> AgentOutput:
        """
        The strongly typed execution logic that all subclasses must implement.
        """
        raise NotImplementedError("Subclasses must implement execute_pydantic")

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overrides AgentBase.execute to enforce Pydantic parsing and validation.
        """
        # Attempt to parse input into AgentInput
        input_data = None
        is_standard_mode = False

        if len(args) == 1 and isinstance(args[0], AgentInput):
            input_data = args[0]
            is_standard_mode = True
        elif len(args) == 1 and isinstance(args[0], str):
            input_data = AgentInput(query=args[0], context=kwargs)
        elif len(args) == 1 and isinstance(args[0], dict):
            # Map common legacy keys
            query = args[0].get("query") or args[0].get("company_id") or args[0].get("company_name") or str(args[0])
            context = args[0].get("context", kwargs)
            input_data = AgentInput(query=query, context=context)
        elif kwargs:
            query = kwargs.get("query") or kwargs.get("company_id") or kwargs.get("company_name") or ""
            input_data = AgentInput(query=query, context=kwargs)
        else:
            logger.error(f"Agent {self.name} received unparseable input args: {args}, kwargs: {kwargs}")
            err_out = AgentOutput(
                answer="Invalid input format provided to agent.",
                confidence=0.0,
                metadata={"error": "Invalid input format", "args": str(args), "kwargs": str(kwargs)}
            )
            return err_out if is_standard_mode else err_out.metadata

        # Set context from input
        if input_data.context:
            self.set_context(input_data.context)
            
        try:
            # Delegate to strictly typed abstract method
            output: AgentOutput = await self.execute_pydantic(input_data)
            
            # Ensure return type is correct
            if not isinstance(output, AgentOutput):
                logger.warning(f"Agent {self.name} returned {type(output)} instead of AgentOutput.")
                if isinstance(output, dict):
                    output = AgentOutput(**output)
                else:
                    raise ValueError(f"Agent {self.name} did not return AgentOutput or valid dict.")
                    
            if not is_standard_mode:
                return output.metadata
            return output
            
        except ValidationError as ve:
            logger.error(f"Pydantic Validation Error in {self.name}: {ve}")
            err_out = AgentOutput(
                answer=f"Output validation failed: {ve}",
                confidence=0.0,
                metadata={"error": str(ve)}
            )
            return err_out if is_standard_mode else err_out.metadata
        except Exception as e:
            logger.exception(f"Execution Error in {self.name}: {e}")
            err_out = AgentOutput(
                answer=f"Execution failed: {e}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
            return err_out if is_standard_mode else err_out.metadata

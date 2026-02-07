from enum import Enum
from typing import List, Dict, Set, Optional

class Role(str, Enum):
    ADMIN = "ADMIN"
    PORTFOLIO_MANAGER = "PORTFOLIO_MANAGER"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"
    GUEST = "GUEST"

class Permission(str, Enum):
    # System
    ADMIN_SYSTEM = "admin:system"
    VIEW_AUDIT_LOGS = "view:audit_logs"

    # Agents
    EXECUTE_AGENT = "agent:execute"
    CONFIGURE_AGENT = "agent:configure"

    # Data
    VIEW_SENSITIVE_DATA = "data:view_sensitive"
    INGEST_DATA = "data:ingest"
    EXPORT_DATA = "data:export"

    # Market
    EXECUTE_TRADE = "market:trade"
    VIEW_MARKET_DATA = "market:view"

class PermissionManager:
    """
    Centralized Permission Manager for RBAC.
    """

    # Default Role -> Permissions Mapping
    ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.ADMIN: {
            Permission.ADMIN_SYSTEM,
            Permission.VIEW_AUDIT_LOGS,
            Permission.EXECUTE_AGENT,
            Permission.CONFIGURE_AGENT,
            Permission.VIEW_SENSITIVE_DATA,
            Permission.INGEST_DATA,
            Permission.EXPORT_DATA,
            Permission.EXECUTE_TRADE,
            Permission.VIEW_MARKET_DATA
        },
        Role.PORTFOLIO_MANAGER: {
            Permission.EXECUTE_AGENT,
            Permission.VIEW_SENSITIVE_DATA,
            Permission.INGEST_DATA,
            Permission.EXPORT_DATA,
            Permission.EXECUTE_TRADE,
            Permission.VIEW_MARKET_DATA
        },
        Role.ANALYST: {
            Permission.EXECUTE_AGENT,
            Permission.VIEW_SENSITIVE_DATA,
            Permission.INGEST_DATA,
            Permission.VIEW_MARKET_DATA
        },
        Role.VIEWER: {
            Permission.VIEW_MARKET_DATA
        },
        Role.GUEST: set()
    }

    def __init__(self, db_session=None):
        self.db_session = db_session

    def get_role_permissions(self, role: str) -> Set[Permission]:
        """Returns the set of permissions for a given role."""
        try:
            role_enum = Role(role.upper())
            return self.ROLE_PERMISSIONS.get(role_enum, set())
        except ValueError:
            return set()

    def has_permission(self, role: str, permission: Permission) -> bool:
        """Checks if a role has a specific permission."""
        permissions = self.get_role_permissions(role)
        return permission in permissions

    def get_user_role(self, user_id: int) -> str:
        """
        Fetches the role for a user.
        In a real scenario, this queries the DB.
        For now, we mock it or use the DB session if provided.
        """
        if self.db_session:
            # Lazy import to avoid circular dependency
            from services.webapp.api import User
            user = self.db_session.query(User).get(user_id)
            if user:
                return user.role
        return Role.GUEST.value

    def enforce(self, user_role: str, permission: Permission) -> bool:
        """
        Enforces a permission check. Raises PermissionError if failed.
        """
        if not self.has_permission(user_role, permission):
            raise PermissionError(f"Access denied: Role '{user_role}' lacks permission '{permission.value}'")
        return True

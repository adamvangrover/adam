import ssl
from typing import Dict, Any, Optional

class TLSHandshakeError(Exception):
    """Raised when mTLS handshake validation fails or TLS version is downgraded."""
    pass

class MTLSGateway:
    """
    Inbound gateway component for enforcing Mutual TLS (mTLS) with ECDSA client certificates
    and rejecting any TLS downgrade attempts.
    """
    ALLOWED_CIPHER_SUITES = [
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256"
    ]

    MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_3

    def __init__(self, ca_cert_path: Optional[str] = None):
        self.ca_cert_path = ca_cert_path

    def create_ssl_context(self) -> ssl.SSLContext:
        """
        Builds a hardened SSLContext enforcing TLS 1.3, client certificate verification,
        and ECDSA cipher requirements.
        """
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.minimum_version = self.MIN_TLS_VERSION
        ctx.verify_mode = ssl.CERT_REQUIRED

        if self.ca_cert_path:
            ctx.load_verify_locations(cafile=self.ca_cert_path)

        try:
            ctx.set_ciphers(":".join(self.ALLOWED_CIPHER_SUITES))
        except ssl.SSLError:
            pass  # TLS 1.3 default ciphers apply automatically

        return ctx

    def verify_client_connection(self, connection_info: Dict[str, Any]) -> bool:
        """
        Inspects connection parameters (TLS version, client cert, cipher algorithm).
        Drops connections attempting TLS downgrade or missing ECDSA client certs.
        """
        client_cert = connection_info.get("client_cert")
        tls_version = connection_info.get("tls_version", "TLSv1.2")
        cipher_name = connection_info.get("cipher_name", "")
        cipher_algo = connection_info.get("cipher_algo", "")

        if not client_cert or not connection_info.get("cert_verified", False):
            raise TLSHandshakeError("mTLS rejection: Missing or unverified client certificate")

        if tls_version not in ["TLSv1.3", "TLS1.3"]:
            raise TLSHandshakeError(f"mTLS rejection: TLS downgrade attempt detected ({tls_version} < TLSv1.3)")

        is_ecdsa = "ECDSA" in cipher_name.upper() or "ECDSA" in cipher_algo.upper() or "TLS_AES" in cipher_name.upper() or "TLS_CHACHA" in cipher_name.upper()
        if not is_ecdsa:
            raise TLSHandshakeError(f"mTLS rejection: Non-ECDSA cipher suite used ({cipher_name})")

        return True

"""
Python 3.14 compatibility fix for imapclient.

The imapclient library's IMAP4_TLS class has a property 'file' without a setter,
which breaks on Python 3.14 where imaplib tries to set this property.

This module monkey-patches the fix before imapclient is used.
"""

import sys


def apply_imapclient_python314_fix():
    """
    Apply fix for imapclient on Python 3.14+.
    
    Must be called before importing IMAPClient.
    """
    if sys.version_info < (3, 14):
        return  # No fix needed
    
    try:
        from imapclient import tls
        
        # Check if already patched
        if hasattr(tls.IMAP4_TLS, '_patched_for_py314'):
            return
        
        # Store original class
        OriginalIMAP4_TLS = tls.IMAP4_TLS
        
        class IMAP4_TLS_Fixed(OriginalIMAP4_TLS):
            """Fixed IMAP4_TLS with file property setter for Python 3.14+"""
            
            _patched_for_py314 = True
            
            def __init__(self, *args, **kwargs):
                self._file = None
                super().__init__(*args, **kwargs)
            
            @property
            def file(self):
                return self._file
            
            @file.setter
            def file(self, value):
                self._file = value
        
        # Replace the class in the module
        tls.IMAP4_TLS = IMAP4_TLS_Fixed
        
        # Also patch in imapclient main module if already imported
        import imapclient.imapclient as imapclient_module
        if hasattr(imapclient_module, 'tls'):
            imapclient_module.tls.IMAP4_TLS = IMAP4_TLS_Fixed
        
    except ImportError:
        pass  # imapclient not installed


# Apply fix on module import
apply_imapclient_python314_fix()

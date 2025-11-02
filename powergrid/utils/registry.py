from typing import Dict, Type, Optional

class ProviderRegistry:
    _types: Dict[str, Type] = {}

    @classmethod
    def register(cls, typ: Type, name: Optional[str] = None) -> None:
        key = name or typ.__name__
        cls._types[key] = typ

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        return cls._types.get(name)

    @classmethod
    def all(cls) -> Dict[str, Type]:
        return dict(cls._types)

def provider(name: Optional[str] = None):
    """Decorator to auto-register provider."""
    def deco(typ):
        ProviderRegistry.register(typ, name)
        return typ
    return deco

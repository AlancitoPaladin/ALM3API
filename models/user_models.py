from typing import Optional

from pydantic import BaseModel


class UserModel(BaseModel):
    name: str
    lastName: str
    secondName: Optional[str] = None
    email: str
    password: str
    role: str
    profilePicture: Optional[str] = None
    bio: Optional[str] = None
    isActive: bool = True

    class Config:
        from_attributes = True

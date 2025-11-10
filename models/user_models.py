from pydantic import BaseModel, EmailStr
from typing import Optional

class UserModel(BaseModel):
    name: str
    lastName: str
    email: str
    password: str
    role: str
    profilePicture: Optional[str] = None
    bio: Optional[str] = None
    isActive: bool = True

    class Config:
        from_attributes = True
from typing import Any, Dict

from pymongo import MongoClient


class MongoDocumentStore:
    def __init__(self, uri: str, db: str, collection: str) -> None:
        self.client = MongoClient(uri)
        self.collection = self.client[db][collection]

    def save_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> None:
        self.collection.update_one(
            {"document_id": document_id},
            {"$set": {"content": content, "metadata": metadata}},
            upsert=True,
        )

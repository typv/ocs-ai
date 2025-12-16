import chromadb
from chromadb import Client
from typing import Optional, Any

class ChromaDBConnector:
    """
    Manages the connection and collection interactions for ChromaDB.

    Supports two connection modes:
    1. Remote (Client-Server): Uses chromadb.HttpClient
    2. Persistent (Local): Uses chromadb.PersistentClient
    """

    def __init__(self, mode: str = "remote", host: str = "localhost", port: int = 8000, path: str = "./chroma_data"):
        """
        Initializes the ChromaDBConnector.

        Args:
            mode (str): The connection mode ('remote' or 'persistent'). Defaults to 'remote'.
            host (str): The host of the ChromaDB Server (only used when mode='remote').
            port (int): The port of the ChromaDB Server (only used when mode='remote').
            path (str): The local data storage path (only used when mode='persistent').
        """
        self.mode = mode
        self.client: Optional[Client] = None

        try:
            if self.mode == "remote":
                # Connect to a remote ChromaDB Server
                self.client = chromadb.HttpClient(host=host, port=port)
                print(f"Successfully connected to Remote ChromaDB at {host}:{port}.")
            elif self.mode == "persistent":
                # Connect to a local Persistent Client
                self.client = chromadb.PersistentClient(path=path)
                print(f"Successfully initialized Persistent Client at {path}.")
            else:
                raise ValueError("Mode must be 'remote' or 'persistent'.")

        except Exception as e:
            print(f"Error initializing ChromaDB Client ({self.mode}): {e}, {host}:{port}")
            self.client = None

    def is_connected(self) -> bool:
        """Checks if the Client has been successfully initialized."""
        return self.client is not None

    def get_collection(self, collection_name: str, embedding_function: Optional[Any] = None) -> Optional[
        chromadb.api.models.Collection.Collection]:
        """
        Retrieves an existing Collection or creates a new one.

        Args:
            collection_name (str): The name of the Collection.
            embedding_function: The Embedding function to use (optional).

        Returns:
            Optional[chromadb.api.models.Collection.Collection]: The ChromaDB Collection object, or None on failure.
        """
        if not self.is_connected():
            print("Error: Client is not connected.")
            return None

        try:
            # get_or_create_collection automatically creates if it doesn't exist
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"Collection **{collection_name}** is ready for use.")
            return collection
        except InvalidCollectionName as e:
            print(f"Invalid Collection Name Error")
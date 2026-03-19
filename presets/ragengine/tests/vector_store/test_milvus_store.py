# Copyright (c) KAITO authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from ragengine.models import Document
from ragengine.tests.vector_store.test_base_store import BaseVectorStoreTest
from ragengine.vector_store.milvus_store import MilvusVectorStoreHandler


class TestMilvusVectorStore(BaseVectorStoreTest):
    """Test implementation for Milvus vector store (Milvus Lite embedded mode)."""

    @pytest.fixture
    def vector_store_manager(self, init_embed_manager):
        with TemporaryDirectory() as temp_dir:
            print(f"Saving temporary test storage at: {temp_dir}")
            os.environ["PERSIST_DIR"] = temp_dir
            # Uses Milvus Lite (embedded) mode — no external server needed
            yield MilvusVectorStoreHandler(init_embed_manager)

    @pytest.mark.asyncio
    async def check_indexed_documents(self, vector_store_manager):
        expected_output_1 = [
            Document(
                doc_id="",
                text="First document in index1",
                metadata={"type": "text"},
                hash_value="81bedde64ebbcd5217992ff7d90fac992c4d7a654e72e76cf5e61c7d45e59afe",
                is_truncated=False,
            )
        ]
        expected_output_2 = [
            Document(
                doc_id="",
                text="First document in index2",
                metadata={"type": "text"},
                hash_value="14f429304e79db9825c4e221723cb90d065978c10972af3a2479de1305e9219d",
                is_truncated=False,
            )
        ]

        for index, expected_output in zip(
            ["index1", "index2"], [expected_output_1, expected_output_2], strict=False
        ):
            resp = await vector_store_manager.list_documents_in_index(
                index, limit=10, offset=0, max_text_length=1000
            )

            assert all(
                resp_doc.text == expected_doc.text
                and resp_doc.hash_value == expected_doc.hash_value
                and resp_doc.metadata == expected_doc.metadata
                for resp_doc, expected_doc in zip(resp.documents, expected_output)
            )

    @property
    def expected_query_score(self):
        """Milvus uses COSINE similarity, scores differ from FAISS L2.

        Set to None to skip exact score comparison.
        """
        return None

    # ── Skip tests that are incompatible with Milvus ────────

    # Milvus update uses delete+re-insert pattern (like Qdrant).
    @pytest.mark.skip(
        reason="Milvus update uses delete+re-insert; see test_update_document_milvus"
    )
    async def test_update_document(self, mock_get, vector_store_manager):
        pass

    # Milvus list order may differ from insertion order.
    @pytest.mark.skip(
        reason="Milvus query order may differ from insertion order; see test_add_document_on_existing_index_milvus"
    )
    async def test_add_document_on_existing_index(self, vector_store_manager):
        pass

    @pytest.mark.skip(reason="Milvus persists to its own embedded DB, not filesystem")
    async def test_persist_and_load_as_seperate_index(self, vector_store_manager):
        pass

    # ── Milvus-specific test overrides ────────────────────────

    @pytest.mark.asyncio
    @patch("requests.get")
    async def test_update_document_milvus(
        self, mock_get, vector_store_manager
    ):
        """Milvus update: delete-then-reinsert."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [{"id": "mock-model", "max_model_len": 2048}]
        }

        documents = [Document(text="Fifth document", metadata={"type": "text"})]
        ids = await vector_store_manager.index_documents("test_index", documents)

        # Update with new text
        result = await vector_store_manager.update_documents(
            "test_index",
            [
                Document(
                    doc_id=ids[0],
                    text="Updated Fifth document",
                    metadata={"type": "text"},
                )
            ],
        )
        assert result["updated_documents"][0].doc_id == ids[0]

        # Verify the doc exists
        assert await vector_store_manager.document_exists(
            "test_index",
            Document(text="Updated Fifth document", metadata={"type": "text"}),
            ids[0],
        )

        # Not-found case
        result = await vector_store_manager.update_documents(
            "test_index",
            [
                Document(
                    doc_id="baddocid",
                    text="Some text",
                    metadata={"type": "text"},
                )
            ],
        )
        assert result["not_found_documents"][0].doc_id == "baddocid"

    @pytest.mark.asyncio
    async def test_add_document_on_existing_index_milvus(self, vector_store_manager):
        """Milvus-specific: verify doc count and doc_id set (not order)."""
        await vector_store_manager.index_documents(
            "test_add_index",
            [Document(text="Initial Doc", metadata={"type": "text"})],
        )

        documents = [
            Document(text=f"Document {i}", metadata={"type": "text"}) for i in range(10)
        ]
        ids = await vector_store_manager.index_documents("test_add_index", documents)

        resp = await vector_store_manager.list_documents_in_index(
            "test_add_index", limit=100, offset=0
        )
        # Should have 11 docs total (1 initial + 10 appended)
        assert resp.total_items >= 11

        # Verify all doc_ids from the second batch are present (order-independent)
        resp_doc_ids = {doc.doc_id for doc in resp.documents}
        assert all(doc_id in resp_doc_ids for doc_id in ids)

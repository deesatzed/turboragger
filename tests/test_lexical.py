import unittest

from turboragger.lexical import BM25PrfRetriever, BM25Retriever, select_expansion_terms, tokenize


class LexicalRetrieverTests(unittest.TestCase):
    def test_tokenize_lowercases_and_keeps_terms(self):
        self.assertEqual(tokenize("Drug-induced Kidney Injury"), ["drug", "induced", "kidney", "injury"])

    def test_bm25_retriever_returns_matching_document_first(self):
        retriever = BM25Retriever(
            {
                "d1": {"title": "cardiology", "text": "heart rhythm procedure"},
                "d2": {"title": "kidney", "text": "renal injury medication"},
            }
        )

        results = retriever.retrieve("renal medication", k=2)

        self.assertEqual(results[0].doc_id, "d2")

    def test_bm25_title_field_ignores_body_only_matches(self):
        retriever = BM25Retriever(
            {
                "d1": {"title": "renal medication", "text": "heart rhythm procedure"},
                "d2": {"title": "cardiology", "text": "renal medication"},
                "d3": {"title": "oncology", "text": "therapy"},
            },
            field="title",
        )

        results = retriever.retrieve("renal medication", k=3)

        self.assertEqual(results[0].doc_id, "d1")
        self.assertGreater(results[0].score, results[1].score)

    def test_select_expansion_terms_uses_feedback_documents_without_query_terms(self):
        terms = select_expansion_terms(
            query_tokens=["renal"],
            feedback_token_lists=[
                ["renal", "injury", "injury", "trial"],
                ["renal", "medication", "injury"],
            ],
            max_terms=2,
        )

        self.assertEqual(terms, ["injury", "medication"])

    def test_prf_retriever_expands_query_from_initial_top_documents(self):
        retriever = BM25PrfRetriever(
            {
                "d1": {"title": "renal overview", "text": "kidney injury kidney injury medication"},
                "d2": {"title": "therapy", "text": "injury medication dosing"},
                "d3": {"title": "cardiology", "text": "heart rhythm procedure"},
            },
            feedback_docs=1,
            expansion_terms=1,
        )

        expanded = retriever.expanded_query("renal")
        results = retriever.retrieve("renal", k=3)

        self.assertIn("injury", expanded)
        self.assertEqual(results[0].doc_id, "d1")


if __name__ == "__main__":
    unittest.main()

"""
Question Classifier for Medical Research Assistant

This module classifies user questions into 'summarization' or 'retrieval' categories
using keyword matching and semantic similarity with sentence transformers.

Compatible with: Python 3.10, 16GB RAM, Intel i7-13th Gen (CPU only)
Dependencies: sentence-transformers==2.2.2

Author: Medical Research Assistant
"""

import re
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sentence-transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class QuestionType(Enum):
    """Enumeration for question types"""
    SUMMARIZATION = "summarization"
    RETRIEVAL = "retrieval"

@dataclass
class ClassificationResult:
    """Result of question classification"""
    question_type: QuestionType
    confidence: float
    method: str
    reasoning: str
    keywords_found: List[str]

class QuestionClassifier:
    """
    Classifies user questions into summarization or retrieval categories
    using keyword matching and semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", enable_semantic: bool = True):
        """
        Initialize the question classifier.
        
        Args:
            model_name: Name of the sentence transformer model
            enable_semantic: Whether to use semantic classification as fallback
        """
        self.model_name = model_name
        self.enable_semantic = enable_semantic
        self.model = None
        self.reference_embeddings = None
        
        # Define keyword patterns for classification
        self.keyword_patterns = {
            QuestionType.SUMMARIZATION: {
                'primary': [
                    r'\bsummariz[e|ing|ed]\b',
                    r'\boverview\b',
                    r'\bsummary\b',
                    r'\bmain\s+findings?\b',
                    r'\bkey\s+points?\b',
                    r'\bhighlights?\b',
                    r'\babstract\b',
                    r'\bconclusions?\b',
                    r'\bresults?\s+across\b',
                    r'\boverall\s+findings?\b',
                    r'\bgeneral\s+findings?\b',
                    r'\bsynthesize\b',
                    r'\bcompile\b',
                    r'\baggregate\b',
                    r'\bcombine\s+findings?\b'
                ],
                'secondary': [
                    r'\bmain\b',
                    r'\bkey\b',
                    r'\bimportant\b',
                    r'\bsignificant\b',
                    r'\bcrucial\b',
                    r'\bessential\b',
                    r'\bprimary\b',
                    r'\bmajor\b'
                ]
            },
            QuestionType.RETRIEVAL: {
                'primary': [
                    r'\bwhat\s+is\b',
                    r'\bwhat\s+are\b',
                    r'\bwhat\s+was\b',
                    r'\bwhat\s+were\b',
                    r'\bhow\s+much\b',
                    r'\bhow\s+many\b',
                    r'\bwhen\s+was\b',
                    r'\bwhen\s+were\b',
                    r'\bwhere\s+is\b',
                    r'\bwhere\s+are\b',
                    r'\bwho\s+is\b',
                    r'\bwho\s+are\b',
                    r'\bwhich\b',
                    r'\bdosage\b',
                    r'\bdose\b',
                    r'\bmedication\b',
                    r'\bdrug\b',
                    r'\btreatment\b',
                    r'\btherapy\b',
                    r'\bprocedure\b',
                    r'\bmethod\b',
                    r'\btechnique\b',
                    r'\bprotocol\b',
                    r'\bspecific\b',
                    r'\bexact\b',
                    r'\bparticular\b',
                    r'\bdetailed?\b',
                    r'\bprecise\b',
                    r'\bin\s+doc\b',
                    r'\bin\s+document\b',
                    r'\bin\s+paper\b',
                    r'\bin\s+study\b',
                    r'\bfrom\s+doc\b',
                    r'\bfrom\s+document\b',
                    r'\baccording\s+to\b',
                    r'\bmentioned\s+in\b',
                    r'\bstated\s+in\b',
                    r'\bfound\s+in\b'
                ],
                'secondary': [
                    r'\bfind\b',
                    r'\bsearch\b',
                    r'\blocate\b',
                    r'\bidentify\b',
                    r'\bextract\b',
                    r'\bretrieve\b',
                    r'\bget\b',
                    r'\bshow\b',
                    r'\btell\b',
                    r'\bexplain\b',
                    r'\bdescribe\b',
                    r'\bdefine\b'
                ]
            }
        }
        
        # Reference questions for semantic classification
        self.reference_questions = {
            QuestionType.SUMMARIZATION: [
                "Summarize the main findings of this research",
                "What are the key points across all documents",
                "Give me an overview of the results",
                "Provide a summary of the conclusions",
                "What are the main highlights from these papers",
                "Synthesize the findings from multiple studies",
                "What are the overall results across documents"
            ],
            QuestionType.RETRIEVAL: [
                "What is the dosage mentioned in document 2",
                "What specific treatment was used in the study",
                "How many patients were included in the trial",
                "What medication was administered to patients",
                "When was this study conducted",
                "Which method was used for analysis",
                "What are the specific side effects mentioned",
                "Find the exact dosage information"
            ]
        }
        
        # Initialize the model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model with error handling"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.enable_semantic:
            print("Semantic classification disabled - using keyword-based classification only")
            return
            
        try:
            print(f"Loading sentence transformer model: {self.model_name}")
            print("This may take a moment on first run...")
            
            # Configure for CPU usage to optimize for your hardware
            device = 'cpu'  # Force CPU usage for compatibility
            
            # Load model with specific configurations for efficiency
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder='./.sentence_transformers_cache'
            )
            
            # Set model to evaluation mode and disable gradient computation
            self.model.eval()
            if hasattr(torch, 'set_grad_enabled'):
                torch.set_grad_enabled(False)
            
            # Pre-compute embeddings for reference questions
            self._precompute_reference_embeddings()
            
            print(f"Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            print("Falling back to keyword-based classification only")
            self.model = None
            self.enable_semantic = False
    
    def _precompute_reference_embeddings(self):
        """Pre-compute embeddings for reference questions to improve performance"""
        if self.model is None:
            return
            
        try:
            print("Pre-computing reference embeddings...")
            
            self.reference_embeddings = {}
            
            for question_type, questions in self.reference_questions.items():
                embeddings = self.model.encode(
                    questions,
                    batch_size=8,  # Smaller batch size for memory efficiency
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                # Calculate mean embedding for each category
                self.reference_embeddings[question_type] = np.mean(embeddings, axis=0)
            
            print("Reference embeddings computed successfully")
            
        except Exception as e:
            print(f"Error computing reference embeddings: {e}")
            self.reference_embeddings = None
    
    def _validate_input(self, question: str) -> bool:
        """Validate input question"""
        if not isinstance(question, str):
            return False
        if len(question.strip()) == 0:
            return False
        if len(question) > 1000:  # Reasonable length limit
            return False
        return True
    
    def _classify_by_keywords(self, question: str) -> Tuple[QuestionType, float, List[str]]:
        """
        Classify question using keyword matching
        
        Args:
            question: Input question string
            
        Returns:
            Tuple of (question_type, confidence, keywords_found)
        """
        question_lower = question.lower()
        
        # Track matches for each category
        matches = {
            QuestionType.SUMMARIZATION: {'primary': [], 'secondary': []},
            QuestionType.RETRIEVAL: {'primary': [], 'secondary': []}
        }
        
        # Check for keyword matches
        for question_type, patterns in self.keyword_patterns.items():
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, question_lower):
                        matches[question_type][pattern_type].append(pattern)
        
        # Calculate scores
        scores = {}
        for question_type, type_matches in matches.items():
            primary_score = len(type_matches['primary']) * 3  # Primary keywords weighted more
            secondary_score = len(type_matches['secondary']) * 1
            scores[question_type] = primary_score + secondary_score
        
        # Determine result
        if scores[QuestionType.SUMMARIZATION] > scores[QuestionType.RETRIEVAL]:
            result_type = QuestionType.SUMMARIZATION
            confidence = min(0.9, 0.6 + (scores[QuestionType.SUMMARIZATION] * 0.1))
            keywords_found = (matches[QuestionType.SUMMARIZATION]['primary'] + 
                            matches[QuestionType.SUMMARIZATION]['secondary'])
        elif scores[QuestionType.RETRIEVAL] > scores[QuestionType.SUMMARIZATION]:
            result_type = QuestionType.RETRIEVAL
            confidence = min(0.9, 0.6 + (scores[QuestionType.RETRIEVAL] * 0.1))
            keywords_found = (matches[QuestionType.RETRIEVAL]['primary'] + 
                            matches[QuestionType.RETRIEVAL]['secondary'])
        else:
            # Equal scores or no matches - default to retrieval
            result_type = QuestionType.RETRIEVAL
            confidence = 0.5
            keywords_found = []
        
        return result_type, confidence, keywords_found
    
    def _classify_by_semantics(self, question: str) -> Tuple[QuestionType, float]:
        """
        Classify question using semantic similarity
        
        Args:
            question: Input question string
            
        Returns:
            Tuple of (question_type, confidence)
        """
        if self.model is None or self.reference_embeddings is None:
            return QuestionType.RETRIEVAL, 0.5
        
        try:
            # Encode the question
            question_embedding = self.model.encode(
                [question],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
            
            # Calculate similarities
            similarities = {}
            for question_type, ref_embedding in self.reference_embeddings.items():
                similarity = np.dot(question_embedding, ref_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(ref_embedding)
                )
                similarities[question_type] = similarity
            
            # Determine result
            if similarities[QuestionType.SUMMARIZATION] > similarities[QuestionType.RETRIEVAL]:
                result_type = QuestionType.SUMMARIZATION
                confidence = min(0.9, similarities[QuestionType.SUMMARIZATION])
            else:
                result_type = QuestionType.RETRIEVAL
                confidence = min(0.9, similarities[QuestionType.RETRIEVAL])
            
            return result_type, max(0.5, confidence)
            
        except Exception as e:
            print(f"Error in semantic classification: {e}")
            return QuestionType.RETRIEVAL, 0.5
    
    def classify_question(self, question: str) -> ClassificationResult:
        """
        Classify a question into summarization or retrieval category
        
        Args:
            question: Input question string
            
        Returns:
            ClassificationResult object with classification details
        """
        # Input validation
        if not self._validate_input(question):
            return ClassificationResult(
                question_type=QuestionType.RETRIEVAL,
                confidence=0.5,
                method="fallback",
                reasoning="Invalid input - defaulting to retrieval",
                keywords_found=[]
            )
        
        try:
            # Primary classification using keywords
            keyword_type, keyword_confidence, keywords_found = self._classify_by_keywords(question)
            
            # If keyword classification is confident, use it
            if keyword_confidence >= 0.7:
                return ClassificationResult(
                    question_type=keyword_type,
                    confidence=keyword_confidence,
                    method="keyword",
                    reasoning=f"Strong keyword match: {', '.join(keywords_found[:3])}",
                    keywords_found=keywords_found
                )
            
            # If semantic classification is available, use it as fallback
            if self.enable_semantic and self.model is not None:
                semantic_type, semantic_confidence = self._classify_by_semantics(question)
                
                # Combine keyword and semantic results
                if keyword_type == semantic_type:
                    # Both methods agree - high confidence
                    combined_confidence = min(0.95, (keyword_confidence + semantic_confidence) / 2 + 0.1)
                    return ClassificationResult(
                        question_type=keyword_type,
                        confidence=combined_confidence,
                        method="keyword + semantic",
                        reasoning=f"Both methods agree. Keywords: {', '.join(keywords_found[:2])}",
                        keywords_found=keywords_found
                    )
                else:
                    # Methods disagree - use semantic if more confident
                    if semantic_confidence > keyword_confidence:
                        return ClassificationResult(
                            question_type=semantic_type,
                            confidence=semantic_confidence,
                            method="semantic",
                            reasoning="Semantic analysis override",
                            keywords_found=keywords_found
                        )
                    else:
                        return ClassificationResult(
                            question_type=keyword_type,
                            confidence=keyword_confidence,
                            method="keyword",
                            reasoning=f"Keyword analysis. Found: {', '.join(keywords_found[:2])}",
                            keywords_found=keywords_found
                        )
            
            # Fallback to keyword result
            return ClassificationResult(
                question_type=keyword_type,
                confidence=keyword_confidence,
                method="keyword",
                reasoning=f"Keyword analysis. Found: {', '.join(keywords_found[:2])}",
                keywords_found=keywords_found
            )
            
        except Exception as e:
            print(f"Error in question classification: {e}")
            return ClassificationResult(
                question_type=QuestionType.RETRIEVAL,
                confidence=0.5,
                method="fallback",
                reasoning=f"Error occurred - defaulting to retrieval: {str(e)}",
                keywords_found=[]
            )
    
    def batch_classify(self, questions: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple questions at once
        
        Args:
            questions: List of question strings
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for question in questions:
            result = self.classify_question(question)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Union[str, bool]]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "semantic_enabled": self.enable_semantic,
            "model_loaded": self.model is not None,
            "reference_embeddings_computed": self.reference_embeddings is not None,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }

def test_question_classifier():
    """
    Test function to verify classification on sample questions
    """
    print("=" * 60)
    print("TESTING QUESTION CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    print("\n1. Initializing classifier...")
    classifier = QuestionClassifier(enable_semantic=True)
    
    # Display model info
    print(f"\n2. Model Information:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test questions
    test_questions = [
        # Summarization questions
        "Summarize the main findings across these papers",
        "What are the key points from all documents?",
        "Give me an overview of the research results",
        "Provide a summary of the conclusions",
        "What are the main highlights from these studies?",
        "Synthesize the findings from multiple papers",
        "What are the overall results across all documents?",
        
        # Retrieval questions
        "What dosage is mentioned in document 2?",
        "What specific treatment was used in the study?",
        "How many patients were included in the trial?",
        "What medication was administered to patients?",
        "When was this study conducted?",
        "Which method was used for data analysis?",
        "What are the specific side effects mentioned?",
        "Find the exact dosage information in paper 3",
        
        # Ambiguous questions
        "What is important in this research?",
        "Tell me about the study",
        "Explain the results",
        "What should I know?"
    ]
    
    print(f"\n3. Testing {len(test_questions)} questions...")
    print("-" * 60)
    
    # Classify all questions
    results = classifier.batch_classify(test_questions)
    
    # Display results
    for i, (question, result) in enumerate(zip(test_questions, results), 1):
        print(f"\n{i:2d}. Question: {question}")
        print(f"    Type: {result.question_type.value.upper()}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Method: {result.method}")
        print(f"    Reasoning: {result.reasoning}")
        if result.keywords_found:
            print(f"    Keywords: {', '.join(result.keywords_found[:3])}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    summarization_count = sum(1 for r in results if r.question_type == QuestionType.SUMMARIZATION)
    retrieval_count = sum(1 for r in results if r.question_type == QuestionType.RETRIEVAL)
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    print(f"Total questions: {len(results)}")
    print(f"Summarization: {summarization_count}")
    print(f"Retrieval: {retrieval_count}")
    print(f"Average confidence: {avg_confidence:.2f}")
    
    # Method usage
    method_counts = {}
    for result in results:
        method = result.method
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\nMethod usage:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    
    print("\nTest completed successfully!")

def main():
    """Main function for standalone execution"""
    print("Question Classifier for Medical Research Assistant")
    print("=" * 50)
    
    # Run tests
    test_question_classifier()
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE")
    print("=" * 50)
    print("Enter questions to classify (type 'quit' to exit):")
    
    classifier = QuestionClassifier()
    
    while True:
        try:
            user_input = input("\nEnter question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            result = classifier.classify_question(user_input)
            
            print(f"\nClassification Result:")
            print(f"  Type: {result.question_type.value.upper()}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Method: {result.method}")
            print(f"  Reasoning: {result.reasoning}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
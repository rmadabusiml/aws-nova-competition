from deepeval.scorer import Scorer
import warnings

# def score_rouge(rouge_type, llm_output, expected_output):
#     scorer = Scorer()
    
#     score = scorer.rouge_score(
#         prediction=llm_output,
#         target=expected_output,
#         score_type=rouge_type
#     )
    
#     print(score)
#     return score
    
def score_rouge(rouge_type, llm_output, expected_output):
    """Score using Rouge metrics with input validation and type conversion"""
    print(f"Scoring Rouge-{rouge_type}")
    
    # Validate and convert inputs to strings
    if llm_output is None or expected_output is None:
        print("Warning: None input detected")
        return 0.0
    
    # Convert to strings if they're not already
    llm_output_str = str(llm_output) if not isinstance(llm_output, str) else llm_output
    expected_output_str = str(expected_output) if not isinstance(expected_output, str) else expected_output
    
    # Additional validation for empty strings
    if not llm_output_str.strip() or not expected_output_str.strip():
        print("Warning: Empty string input detected after conversion")
        return 0.0
    
    scorer = Scorer()
    
    try:
        score = scorer.rouge_score(
            prediction=llm_output_str,
            target=expected_output_str,
            score_type=rouge_type
        )
        
        print(f"Rouge-{rouge_type} Score: {score:.4f}")
        return score
    except Exception as e:
        print(f"Error calculating Rouge score: {e}")
        return 0.0

def score_bleu(bleu_type, llm_output, expected_output):
    # Validate and convert inputs to strings
    if llm_output is None or expected_output is None:
        print("Warning: None input detected")
        return 0.0
    
    # Convert to strings if they're not already
    llm_output_str = str(llm_output) if not isinstance(llm_output, str) else llm_output
    expected_output_str = str(expected_output) if not isinstance(expected_output, str) else expected_output
    
    # Additional validation for empty strings
    if not llm_output_str.strip() or not expected_output_str.strip():
        print("Warning: Empty string input detected after conversion")
        return 0.0
        
    scorer = Scorer()
    
    # score = scorer.sentence_bleu_score(
    #     prediction=llm_output_str,
    #     references=expected_output_str,
    #     bleu_type=bleu_type
    # )

    try:
        # Suppress BLEU warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            score = scorer.sentence_bleu_score(
                prediction=llm_output_str,
                references=expected_output_str,
                bleu_type=bleu_type
            )
        
        print(f"{bleu_type.upper()} Score: {score:.4f}")
        return score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0
    
    print(score)
    return score

# def score_rouge(rouge_type, llm_output, expected_output):
#     """Score using Rouge metrics with validation"""
#     print(f"Scoring Rouge-{rouge_type}")
    
#     # Validate inputs
#     if not llm_output or not expected_output:
#         print("Warning: Empty input detected")
#         return 0.0
    
#     scorer = Scorer()
    
#     try:
#         score = scorer.rouge_score(
#             prediction=llm_output,
#             target=expected_output,
#             score_type=rouge_type
#         )
#         print(f"Rouge-{rouge_type} Score: {score:.4f}")
#         return score
#     except Exception as e:
#         print(f"Error calculating Rouge score: {e}")
#         return 0.0

# def score_bleu(bleu_type, llm_output, expected_output):
#     """Score using BLEU metrics with smoothing function"""
#     print(f"Scoring {bleu_type.upper()}")
    
#     # Validate inputs
#     if not llm_output or not expected_output:
#         print("Warning: Empty input detected")
#         return 0.0
    
#     scorer = Scorer()
    
#     try:
#         # Suppress BLEU warnings for cleaner output
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
            
#             score = scorer.sentence_bleu_score(
#                 prediction=llm_output,
#                 references=expected_output,
#                 bleu_type=bleu_type
#             )
        
#         print(f"{bleu_type.upper()} Score: {score:.4f}")
#         return score
#     except Exception as e:
#         print(f"Error calculating BLEU score: {e}")
#         return 0.0
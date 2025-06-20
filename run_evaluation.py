#!/usr/bin/env python3
"""
Script to run entity alignment evaluation
"""

from entity_alignment_evaluator import EntityAlignmentEvaluator

def main():
    # File paths
    predictions_file = 'aligned_enitites/entity_alignment.csv'
    ground_truth_file = 'data/DBP15K/torch_geometric_cache/raw/fr_en/train.ref'
    
    print("🚀 Starting Entity Alignment Evaluation...")
    print(f"Predictions file: {predictions_file}")
    print(f"Ground truth file: {ground_truth_file}")
    
    # Initialize evaluator
    evaluator = EntityAlignmentEvaluator(predictions_file, ground_truth_file)
    
    # Run evaluation
    try:
        results = evaluator.run_complete_evaluation()
        
        # Print report
        evaluator.print_evaluation_report()
        
        # Save results
        evaluator.save_detailed_results('entity_alignment_evaluation_results.json')
        
        print('\n✅ Evaluation complete! Check the detailed results in the JSON file.')
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
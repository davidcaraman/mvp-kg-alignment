import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EntityAlignmentEvaluator:
    """
    Comprehensive evaluator for entity alignment results.
    Calculates various metrics including Hits@K, MRR, precision, recall, F1-score,
    and confidence-based analysis.
    """
    
    def __init__(self, predictions_file: str, ground_truth_file: str):
        """
        Initialize the evaluator with prediction and ground truth files.
        
        Args:
            predictions_file: CSV file with columns [index_id, candidate_id, confidence_score, need_judge]
            ground_truth_file: Reference file with ground truth mappings
        """
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file
        self.predictions_df = None
        self.ground_truth_df = None
        self.evaluation_results = {}
        
    def load_data(self):
        """Load and preprocess the prediction and ground truth data."""
        print("Loading data...")
        
        # Load predictions
        self.predictions_df = pd.read_csv(self.predictions_file)
        print(f"Loaded {len(self.predictions_df)} predictions")
        
        # Load ground truth
        self.ground_truth_df = pd.read_csv(
            self.ground_truth_file, 
            sep='\t', 
            names=['index_id', 'true_candidate_id'],
            header=None
        )
        print(f"Loaded {len(self.ground_truth_df)} ground truth mappings")
        
        # Clean and preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Clean and preprocess the loaded data."""
        # Handle missing values in predictions
        # Replace -1 and NaN candidate_ids with None for consistent handling
        self.predictions_df['candidate_id'] = self.predictions_df['candidate_id'].replace(-1, np.nan)
        
        # Ensure data types
        self.predictions_df['index_id'] = self.predictions_df['index_id'].astype(int)
        self.predictions_df['confidence_score'] = pd.to_numeric(self.predictions_df['confidence_score'], errors='coerce')
        
        self.ground_truth_df['index_id'] = self.ground_truth_df['index_id'].astype(int)
        self.ground_truth_df['true_candidate_id'] = self.ground_truth_df['true_candidate_id'].astype(int)
        
        print(f"Preprocessing complete:")
        print(f"  - Predictions with missing candidate_id: {self.predictions_df['candidate_id'].isna().sum()}")
        print(f"  - Total entities in ground truth: {len(self.ground_truth_df)}")
        print(f"  - Total predictions: {len(self.predictions_df)}")
        
    def calculate_hits_at_k(self, k: int = 1) -> float:
        """
        Calculate Hits@K metric.
        For this dataset, we only have top-1 predictions, so this is essentially accuracy.
        
        Args:
            k: The K value for Hits@K (default=1)
            
        Returns:
            Hits@K score as a float
        """
        # Merge predictions with ground truth
        merged_df = pd.merge(
            self.predictions_df, 
            self.ground_truth_df, 
            on='index_id', 
            how='inner'
        )
        
        # Calculate hits@1 (correct predictions)
        correct_predictions = (merged_df['candidate_id'] == merged_df['true_candidate_id'])
        hits_at_k = correct_predictions.sum() / len(merged_df)
        
        return hits_at_k
    
    def calculate_precision_recall_f1(self) -> dict:
        """
        Calculate precision, recall, and F1-score.
        Treats this as a binary classification where we predict whether each entity
        has a match or not.
        """
        # Merge predictions with ground truth
        merged_df = pd.merge(
            self.predictions_df, 
            self.ground_truth_df, 
            on='index_id', 
            how='inner'
        )
        
        # Binary classification: predicted match vs no match
        y_true = np.ones(len(merged_df))  # All ground truth entities have matches
        y_pred = (~merged_df['candidate_id'].isna()).astype(int)  # Predicted to have match
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def calculate_coverage_metrics(self) -> dict:
        """Calculate coverage-related metrics."""
        total_predictions = len(self.predictions_df)
        predictions_with_candidates = (~self.predictions_df['candidate_id'].isna()).sum()
        predictions_need_judge = self.predictions_df['need_judge'].sum()
        
        # Coverage of ground truth
        ground_truth_ids = set(self.ground_truth_df['index_id'])
        predicted_ids = set(self.predictions_df['index_id'])
        coverage = len(predicted_ids.intersection(ground_truth_ids)) / len(ground_truth_ids)
        
        return {
            'total_predictions': total_predictions,
            'predictions_with_candidates': predictions_with_candidates,
            'prediction_coverage': predictions_with_candidates / total_predictions,
            'predictions_need_judge': predictions_need_judge,
            'judge_rate': predictions_need_judge / total_predictions,
            'ground_truth_coverage': coverage
        }
    
    def analyze_confidence_scores(self) -> dict:
        """Analyze the distribution and effectiveness of confidence scores."""
        # Merge with ground truth to analyze confidence vs accuracy
        merged_df = pd.merge(
            self.predictions_df, 
            self.ground_truth_df, 
            on='index_id', 
            how='inner'
        )
        
        # Calculate accuracy by confidence level
        confidence_accuracy = {}
        confidence_counts = {}
        
        for conf in range(1, 6):  # Confidence scores 1-5
            conf_subset = merged_df[merged_df['confidence_score'] == conf]
            if len(conf_subset) > 0:
                # Only consider predictions with candidates
                conf_with_candidates = conf_subset[~conf_subset['candidate_id'].isna()]
                if len(conf_with_candidates) > 0:
                    accuracy = (conf_with_candidates['candidate_id'] == conf_with_candidates['true_candidate_id']).mean()
                    confidence_accuracy[conf] = accuracy
                    confidence_counts[conf] = len(conf_subset)
                else:
                    confidence_accuracy[conf] = 0.0
                    confidence_counts[conf] = len(conf_subset)
        
        return {
            'confidence_distribution': self.predictions_df['confidence_score'].value_counts().to_dict(),
            'confidence_accuracy': confidence_accuracy,
            'confidence_counts': confidence_counts,
            'avg_confidence': self.predictions_df['confidence_score'].mean(),
            'std_confidence': self.predictions_df['confidence_score'].std()
        }
    
    def calculate_error_analysis(self) -> dict:
        """Perform detailed error analysis."""
        merged_df = pd.merge(
            self.predictions_df, 
            self.ground_truth_df, 
            on='index_id', 
            how='inner'
        )
        
        # Types of errors
        no_prediction = merged_df['candidate_id'].isna().sum()
        wrong_prediction = ((~merged_df['candidate_id'].isna()) & 
                           (merged_df['candidate_id'] != merged_df['true_candidate_id'])).sum()
        correct_prediction = ((~merged_df['candidate_id'].isna()) & 
                             (merged_df['candidate_id'] == merged_df['true_candidate_id'])).sum()
        
        # Error by confidence level
        error_by_confidence = {}
        for conf in range(1, 6):
            conf_subset = merged_df[merged_df['confidence_score'] == conf]
            if len(conf_subset) > 0:
                conf_errors = ((~conf_subset['candidate_id'].isna()) & 
                              (conf_subset['candidate_id'] != conf_subset['true_candidate_id'])).sum()
                error_by_confidence[conf] = {
                    'errors': conf_errors,
                    'total': len(conf_subset),
                    'error_rate': conf_errors / len(conf_subset) if len(conf_subset) > 0 else 0
                }
        
        return {
            'no_prediction': no_prediction,
            'wrong_prediction': wrong_prediction,
            'correct_prediction': correct_prediction,
            'total_evaluated': len(merged_df),
            'error_by_confidence': error_by_confidence
        }
    
    def run_complete_evaluation(self) -> dict:
        """Run the complete evaluation and return all metrics."""
        print("Running complete evaluation...")
        
        # Load data
        self.load_data()
        
        # Calculate all metrics
        results = {}
        
        # Primary metrics
        results['hits_at_1'] = self.calculate_hits_at_k(1)
        results['precision_recall_f1'] = self.calculate_precision_recall_f1()
        results['coverage_metrics'] = self.calculate_coverage_metrics()
        results['confidence_analysis'] = self.analyze_confidence_scores()
        results['error_analysis'] = self.calculate_error_analysis()
        
        # Store results
        self.evaluation_results = results
        
        return results
    
    def print_evaluation_report(self):
        """Print a comprehensive evaluation report."""
        if not self.evaluation_results:
            print("No evaluation results available. Run run_complete_evaluation() first.")
            return
        
        results = self.evaluation_results
        
        print("\n" + "="*80)
        print("ENTITY ALIGNMENT EVALUATION REPORT")
        print("="*80)
        
        # Primary Performance Metrics
        print(f"\n🎯 PRIMARY PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Hits@1 (Accuracy):           {results['hits_at_1']:.4f} ({results['hits_at_1']*100:.2f}%)")
        print(f"Precision:                   {results['precision_recall_f1']['precision']:.4f}")
        print(f"Recall:                      {results['precision_recall_f1']['recall']:.4f}")
        print(f"F1-Score:                    {results['precision_recall_f1']['f1_score']:.4f}")
        
        # Coverage Metrics
        print(f"\n📊 COVERAGE METRICS")
        print("-" * 40)
        coverage = results['coverage_metrics']
        print(f"Total Predictions:           {coverage['total_predictions']:,}")
        print(f"Predictions with Candidates: {coverage['predictions_with_candidates']:,}")
        print(f"Prediction Coverage:         {coverage['prediction_coverage']:.4f} ({coverage['prediction_coverage']*100:.2f}%)")
        print(f"Predictions Needing Judge:   {coverage['predictions_need_judge']:,}")
        print(f"Judge Rate:                  {coverage['judge_rate']:.4f} ({coverage['judge_rate']*100:.2f}%)")
        print(f"Ground Truth Coverage:       {coverage['ground_truth_coverage']:.4f} ({coverage['ground_truth_coverage']*100:.2f}%)")
        
        # Confidence Analysis
        print(f"\n🎲 CONFIDENCE SCORE ANALYSIS")
        print("-" * 40)
        conf_analysis = results['confidence_analysis']
        print(f"Average Confidence:          {conf_analysis['avg_confidence']:.2f}")
        print(f"Std Dev Confidence:          {conf_analysis['std_confidence']:.2f}")
        
        print("\nConfidence Distribution:")
        for conf in sorted(conf_analysis['confidence_distribution'].keys()):
            count = conf_analysis['confidence_distribution'][conf]
            percentage = count / coverage['total_predictions'] * 100
            print(f"  Confidence {conf}: {count:,} ({percentage:.1f}%)")
        
        print("\nAccuracy by Confidence Level:")
        for conf in sorted(conf_analysis['confidence_accuracy'].keys()):
            accuracy = conf_analysis['confidence_accuracy'][conf]
            count = conf_analysis['confidence_counts'][conf]
            print(f"  Confidence {conf}: {accuracy:.4f} ({accuracy*100:.2f}%) - {count:,} predictions")
        
        # Error Analysis
        print(f"\n🔍 ERROR ANALYSIS")
        print("-" * 40)
        error_analysis = results['error_analysis']
        total = error_analysis['total_evaluated']
        print(f"Total Evaluated:             {total:,}")
        print(f"Correct Predictions:         {error_analysis['correct_prediction']:,} ({error_analysis['correct_prediction']/total*100:.2f}%)")
        print(f"Wrong Predictions:           {error_analysis['wrong_prediction']:,} ({error_analysis['wrong_prediction']/total*100:.2f}%)")
        print(f"No Predictions:              {error_analysis['no_prediction']:,} ({error_analysis['no_prediction']/total*100:.2f}%)")
        
        print("\nError Rate by Confidence Level:")
        for conf in sorted(error_analysis['error_by_confidence'].keys()):
            error_data = error_analysis['error_by_confidence'][conf]
            print(f"  Confidence {conf}: {error_data['error_rate']:.4f} ({error_data['error_rate']*100:.2f}%) - {error_data['errors']}/{error_data['total']} errors")
        
        print("\n" + "="*80)
    
    def save_detailed_results(self, output_file: str = "evaluation_results.json"):
        """Save detailed evaluation results to a JSON file."""
        import json
        
        if not self.evaluation_results:
            print("No evaluation results to save. Run evaluation first.")
            return
        
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        cleaned_results = convert_numpy_types(self.evaluation_results)
        
        with open(output_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")


def main():
    """Main function to run the evaluation."""
    # File paths
    predictions_file = "aligned_enitites/entity_alignment.csv"
    ground_truth_file = "data/DBP15K/torch_geometric_cache/raw/fr_en/train.ref"
    
    # Initialize evaluator
    evaluator = EntityAlignmentEvaluator(predictions_file, ground_truth_file)
    
    # Run evaluation
    results = evaluator.run_complete_evaluation()
    
    # Print report
    evaluator.print_evaluation_report()
    
    # Save results
    evaluator.save_detailed_results("entity_alignment_evaluation_results.json")
    
    print(f"\n✅ Evaluation complete! Check the detailed results in the JSON file.")
    
    return evaluator


if __name__ == "__main__":
    evaluator = main() 
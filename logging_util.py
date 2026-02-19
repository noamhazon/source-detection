import os
import csv
from datetime import datetime
from config import LOG_DIR, ENABLE_DETAILED_LOGGING

class SimulationLogger:
    """Manages logging of simulation details to CSV files."""
    
    def __init__(self, experiment_name):
        """
        Initialize logger for an experiment.
        
        Args:
            experiment_name: Name of the experiment (used for log filename)
        """
        self.experiment_name = experiment_name
        self.log_entries = []
        self._ensure_log_dir()
        self.log_file = self._get_log_filename()
        self._write_header()
    
    def _ensure_log_dir(self):
        """Create log directory if it doesn't exist."""
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
    
    def _get_log_filename(self):
        """Generate log filename. The timestamp is already embedded within experiment_name"""
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{LOG_DIR}/{self.experiment_name}.csv"
        return filename
    
    def _write_header(self):
        """Write CSV header."""
        headers = [
            'random_seed',
            'timestamp',
            'source_node',
            'size_A',
            'size_A_prime',
            'method',
            'running_time_seconds',
            'success',
            'rank_of_true_source',
            'additional_info'
        ]
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def log_simulation(self, random_seed, source_node, size_A, size_A_prime, method, 
                      running_time, success, rank_of_true_source=None, 
                      additional_info=None):
        """
        Log a simulation result.
        
        Args:
            random_seed: The random seed used for the simulation
            source_node: The true source node
            size_A: Size of diffusion set A
            size_A_prime: Size of candidate set A'
            method: Name of the source detection method
            running_time: Execution time in seconds
            success: Boolean indicating if method succeeded (found true source)
            rank_of_true_source: Rank of true source in predictions (if applicable)
            additional_info: Any additional information to log
        """
        if not ENABLE_DETAILED_LOGGING:
            return
        
        entry = {
            'random_seed': random_seed,
            'timestamp': datetime.now().isoformat(),
            'source_node': source_node,
            'size_A': size_A,
            'size_A_prime': size_A_prime,
            'method': method,
            'running_time_seconds': f"{running_time:.4f}",
            'success': success,
            'rank_of_true_source': rank_of_true_source if rank_of_true_source is not None else 'N/A',
            'additional_info': additional_info if additional_info else ''
        }
        
        self.log_entries.append(entry)
        self._append_to_file(entry)
    
    def _append_to_file(self, entry):
        """Append entry to CSV file."""
        headers = [
            'random_seed',
            'timestamp',
            'source_node',
            'size_A',
            'size_A_prime',
            'method',
            'running_time_seconds',
            'success',
            'rank_of_true_source',
            'additional_info'
        ]
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(entry)
    
    def get_statistics(self):
        """Return summary statistics of logged simulations."""
        if not self.log_entries:
            return {}
        
        total = len(self.log_entries)
        successes = sum(1 for e in self.log_entries if e['success'] == True or e['success'] == 'True')
        
        stats = {
            'total_simulations': total,
            'successful': successes,
            'success_rate': f"{100 * successes / total:.2f}%",
            'log_file': self.log_file
        }
        return stats

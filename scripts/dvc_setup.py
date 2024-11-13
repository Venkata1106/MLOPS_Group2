import os
import subprocess
import logging
from typing import Dict, Any, List, Optional
import yaml
from datetime import datetime

def check_dvc_installation() -> Dict[str, Any]:
    """Check if DVC is installed and install if needed"""
    try:
        # Try to import dvc
        import dvc
        return {
            'status': 'success',
            'message': f'DVC is installed (version {dvc.__version__})'
        }
    except ImportError:
        try:
            self.logger.info("DVC not found. Attempting to install...")
            subprocess.run(
                ['pip', 'install', 'dvc'],
                check=True,
                capture_output=True,
                text=True
            )
            import dvc
            return {
                'status': 'success',
                'message': f'DVC installed successfully (version {dvc.__version__})'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': f'Failed to install DVC: {str(e)}'
            }

class DVCManager:
    """Class for managing DVC setup and operations"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize DVCManager
        
        Parameters:
        -----------
        project_root : str, optional
            Root directory of the project
        """
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Define paths
        self.data_dir = os.path.join(self.project_root, 'data')
        self.dvc_dir = os.path.join(self.project_root, '.dvc')
        self.config_file = os.path.join(self.project_root, 'dvc.yaml')
        
        # Check DVC installation first
        install_check = check_dvc_installation()
        if install_check['status'] == 'error':
            self.logger.error(install_check['error_message'])
            raise RuntimeError("DVC is not installed. Please run: pip install dvc")
    
    def run_command(self, command: List[str]) -> Dict[str, Any]:
        """
        Run a shell command and return results
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return {
                'status': 'success',
                'output': result.stdout,
                'command': ' '.join(command)
            }
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(command)}")
            self.logger.error(f"Error output: {e.stderr}")
            return {
                'status': 'error',
                'error': str(e),
                'command': ' '.join(command)
            }
    
    def initialize_git(self) -> Dict[str, Any]:
        """Initialize Git repository if not already initialized"""
        try:
            git_dir = os.path.join(self.project_root, '.git')
            if not os.path.exists(git_dir):
                self.logger.info("Initializing Git repository...")
                result = self.run_command(['git', 'init'])
                if result['status'] == 'success':
                    # Set up initial Git configuration
                    self.run_command(['git', 'add', '.'])
                    self.run_command(['git', 'commit', '-m', 'Initial commit'])
                return result
            else:
                return {
                    'status': 'success',
                    'message': 'Git repository already initialized'
                }
        except Exception as e:
            self.logger.error(f"Error initializing Git: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def initialize_dvc(self) -> Dict[str, Any]:
        """Initialize DVC in the project"""
        try:
            results = []
            
            # First initialize Git
            git_result = self.initialize_git()
            results.append(git_result)
            
            # Initialize DVC if not already initialized
            if not os.path.exists(self.dvc_dir):
                # Use --no-scm if Git initialization failed
                if git_result['status'] == 'error':
                    results.append(self.run_command(['dvc', 'init', '--no-scm']))
                else:
                    results.append(self.run_command(['dvc', 'init']))
                self.logger.info("DVC initialized successfully")
            
            # Create data directories
            for subdir in ['raw', 'processed', 'analyzed', 'models']:
                dir_path = os.path.join(self.data_dir, subdir)
                os.makedirs(dir_path, exist_ok=True)
            
            return {
                'status': 'success',
                'message': 'DVC initialized successfully',
                'commands': results
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing DVC: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def create_dvc_config(self) -> Dict[str, Any]:
        """Create DVC pipeline configuration"""
        try:
            config = {
                'stages': {
                    'data_acquisition': {
                        'cmd': 'python scripts/data_acquisition.py',
                        'deps': ['scripts/data_acquisition.py'],
                        'outs': ['data/raw'],
                        'metrics': [],
                        'params': ['config/params.yaml:data_acquisition']
                    },
                    'data_preprocessing': {
                        'cmd': 'python scripts/data_preprocessing.py',
                        'deps': ['scripts/data_preprocessing.py', 'data/raw'],
                        'outs': ['data/processed'],
                        'metrics': [],
                        'params': ['config/params.yaml:preprocessing']
                    },
                    'data_validation': {
                        'cmd': 'python scripts/data_validation.py',
                        'deps': ['scripts/data_validation.py', 'data/processed'],
                        'outs': [],
                        'metrics': ['metrics/validation_metrics.json'],
                        'params': ['config/params.yaml:validation']
                    },
                    'bias_detection': {
                        'cmd': 'python scripts/bias_detection.py',
                        'deps': ['scripts/bias_detection.py', 'data/processed'],
                        'outs': [],
                        'metrics': ['metrics/bias_metrics.json'],
                        'params': ['config/params.yaml:bias_detection']
                    }
                }
            }
            
            # Save config
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Created DVC config file: {self.config_file}")
            return {
                'status': 'success',
                'config_path': self.config_file
            }
            
        except Exception as e:
            self.logger.error(f"Error creating DVC config: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def add_data_to_dvc(self, data_path: str) -> Dict[str, Any]:
        """Add data to DVC tracking"""
        try:
            result = self.run_command(['dvc', 'add', data_path])
            if result['status'] == 'success':
                self.logger.info(f"Added {data_path} to DVC tracking")
                
                # Stage changes in git
                self.run_command(['git', 'add', f"{data_path}.dvc"])
                self.run_command(['git', 'add', '.gitignore'])
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error adding data to DVC: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def setup_remote_storage(self, remote_url: str, remote_name: str = 'origin') -> Dict[str, Any]:
        """Set up remote storage for DVC"""
        try:
            commands = [
                ['dvc', 'remote', 'add', '-d', remote_name, remote_url],
                ['dvc', 'remote', 'modify', remote_name, 'verify', 'true']
            ]
            
            results = [self.run_command(cmd) for cmd in commands]
            
            return {
                'status': 'success',
                'remote_name': remote_name,
                'remote_url': remote_url,
                'commands': results
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up remote storage: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }

def setup_dvc(project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to set up DVC
    """
    manager = DVCManager(project_root)
    
    # Check DVC installation
    install_check = check_dvc_installation()
    if install_check['status'] == 'error':
        return install_check
    
    # Run setup steps
    init_result = manager.initialize_dvc()
    config_result = manager.create_dvc_config()
    
    # Create .gitignore if it doesn't exist
    gitignore_path = os.path.join(manager.project_root, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write("/data\n")
            f.write("/.dvc/cache\n")
    
    return {
        'initialization': init_result,
        'configuration': config_result
    }

if __name__ == "__main__":
    # Example usage
    results = setup_dvc()
    
    if results['initialization']['status'] == 'success':
        print("\nDVC setup completed successfully!")
        print("\nNext steps:")
        print("1. Review the dvc.yaml file")
        print("2. Add your data:")
        print("   dvc add data/raw")
        print("3. Push to remote storage (optional):")
        print("   dvc remote add -d storage s3://your-bucket/path")
        print("   dvc push")
    else:
        print("\nDVC setup encountered some issues. Please check the error messages above.")
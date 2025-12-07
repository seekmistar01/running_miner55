## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd subnet-niome

# Install NIOME subnet codebase
pip install -e .

# Run miner example on local network:
python3 neurons/miner.py   --wallet.name miner   --wallet.hotkey hotkey_miner   --netuid 2   --axon.port 8902   --axon.ip 127.0.0.1   --subtensor.network local   --logging.debug


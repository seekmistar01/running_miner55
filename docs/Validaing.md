##  Validator Setup Guide 

To participate as a Validator in the NIOME subnet, you must first set up your machine, install the necessary software, and register your identity on the Bittensor blockchain.

### 1. Prerequisites

Before starting, ensure your system meets the minimum requirements and has the core dependencies installed.

* **Operating System:** Ubuntu 22.04 or similar Linux distribution is generally recommended for optimal compatibility. Mining is not supported on Windows.
* **Python:** Python 3.12 or higher
* **Git:** Necessary for cloning the repository.
* **Hardware:** 4GB RAM, 4 core CPU

### 2. Environment Setup

This section walks you through cloning the NIOME repository and installing the required libraries.

1. **Clone the Repository:**
   **Bash**

   ```
   git clone https://github.com/genomesio/subnet-niome.git
   cd subnet-niome
   ```
2. **Create a Virtual Environment (Recommended):**
   **Bash**

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:** Install the required Python packages and register the local package for execution.
   **Bash**

   ```
   python3 -m pip install -r requirements.txt
   ```

### 3. Wallet Creation and Registration

You must create a Bittensor wallet to hold your TAO and Alpha tokens, and to register your hotkey with the subnet.

1. **Install Bittensor CLI:**
   **Bash**

   ```
   python3 -m pip install bittensor-cli
   ```
2. **Create a Coldkey (Primary Wallet):** The coldkey is your secure, offline store of funds. Choose a secure name, e.g., `niome_wallet`.
   **Bash**

   ```
   btcli wallet new_coldkey --wallet.name niome_wallet
   ```
3. **Create a Hotkey (Validator Identity):** The hotkey is used to sign transactions, run the validator, and receive emissions. It is connected to your coldkey.
   **Bash**

   ```
   btcli wallet new_hotkey --wallet.name niome_wallet --wallet.hotkey niome_validator
   ```
4. **Fund Your Coldkey:** Transfer a small amount of TAO to your coldkey to cover registration fees, which fluctuate based on subnet competition.
5. **Register Your Hotkey to Subnet:** Register your hotkey to secure a UID (Unique Identifier) on the NIOME subnet. The Network ID for NIOME is xx.
   **Bash**

   ```
   btcli subnet register --netuid xx --wallet.name niome_wallet --wallet.hotkey niome_validator
   ```
    In case of testnet

    ```
   btcli subnet register --netuid 289 --network test --wallet.name niome_wallet --wallet.hotkey niome_validator
   ```

   *Note: Replace `finney` with `mainnet` if running on the live network, and ensure you have sufficient TAO for the current registration fee.*

### 4. Running the Validator

Once your hotkey is registered, you can start your Validator. The parameters you pass will determine your specific role (Architect, Adversary, or Oracle) and network configuration.

1. **Set the python environment variable:**
   **Bash**

   ```
    export PYTHONPATH = "$PYTHONPATH:~/niome_subnet"
   ```

2. **Run the Validator Script:** The core command to launch a validator neuron requires specifying your wallet and hotkey names, the network, and the subnet ID (`--netuid xx`, `testnet uid: 289`).
   **Bash**

   ```
   pm2 start "python neurons/validator.py --netuid xx --subtensor.network testorfinney --wallet.name niome_wallet --wallet.hotkey niome_validator --logging.debug" --name niome-validator
   ```

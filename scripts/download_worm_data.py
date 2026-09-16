import os
import subprocess
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEST_DIR = "data/worm"
OWM_REPO_URL = "https://github.com/openworm/OpenWormData.git"

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    owm_dir = os.path.join(DEST_DIR, ".owm")
    
    if os.path.exists(owm_dir):
        logger.info(f"OpenWorm database already exists at {owm_dir}. Skipping clone.")
    else:
        logger.info(f"Cloning OpenWorm owmeta database into {DEST_DIR}...")
        try:
            # We run the command inside DEST_DIR so it creates .owm there
            subprocess.run(
                ["owm", "clone", OWM_REPO_URL, "--branch", "owmeta"],
                cwd=DEST_DIR,
                check=True
            )
            logger.info("Successfully cloned OpenWormData.")
        except FileNotFoundError:
            logger.error("The 'owm' command was not found. Please install owmeta (pip install owmeta owmeta-core) and try again.")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone OpenWormData: {e}")
            return

    # Basic connection test to verify it works
    logger.info("Testing connection to OpenWorm database...")
    try:
        from owmeta_core.command import OWM
        from owmeta_core.context import Context
        from owmeta.worm import Worm
        
        # Connect to the .owm directory inside DEST_DIR
        old_cwd = os.getcwd()
        os.chdir(DEST_DIR)
        
        conn = OWM().connect()
        ctx = conn(Context)(ident='http://openworm.org/data')
        
        # Test query: how many muscles?
        muscles = ctx.stored(Worm).query().muscles()
        logger.info(f"Successfully connected! Found {len(muscles)} muscles in the C. elegans database.")
        
        conn.disconnect()
        os.chdir(old_cwd)
        
    except ImportError:
        logger.error("Could not import owmeta. Please install it using 'pip install owmeta owmeta-core'.")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")

if __name__ == "__main__":
    main()

from loguru import logger

from med_edge.dataset_handler.hugging_face_handler import get_med_qa_dataset

if __name__ == "__main__":
    logger.info("Starting single LLM benchmark")
    med_qa = get_med_qa_dataset()
    all = {len(example['options']) for example in med_qa.train}
    print(all)
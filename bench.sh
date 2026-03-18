#!/bin/bash
# Flash Attention benchmark: short vs long context
# Usage: bash bench.sh

BIN="bazel run //bin/qwen:qwen --"
MODEL="/home/hebin/vvllm/models/Qwen2.5-0.5B"
COMMON="--backend cuda --quantize int8 --fp16"

echo "============================================"
echo "  Flash Attention Benchmark"
echo "============================================"
echo ""

echo ">>> Test 1: Short prompt (4 tokens), 50 decode tokens"
echo "--------------------------------------------"
$BIN --model $MODEL $COMMON \
    --prompt "Once upon a time" \
    --max_tokens 50
echo ""

echo ">>> Test 2: Medium prompt (64 tokens), 200 decode tokens"
echo "--------------------------------------------"
$BIN --model $MODEL $COMMON \
    --prompt "In the year 2045, humanity had finally achieved what scientists once thought impossible. The first self-sustaining colony on Mars was not just surviving but thriving. Dr. Elena Chen stood at the observation deck of Olympus Station, watching the rust-colored landscape stretch endlessly before her. The thin atmosphere scattered light in ways that" \
    --max_tokens 200
echo ""

echo ">>> Test 3: Long prompt (128+ tokens), 300 decode tokens"
echo "--------------------------------------------"
$BIN --model $MODEL $COMMON \
    --prompt "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades." \
    --max_tokens 300
echo ""

echo ">>> Test 4: Very long prompt (256+ tokens), 500 decode tokens"
echo "--------------------------------------------"
$BIN --model $MODEL $COMMON \
    --prompt "Once upon a time in a kingdom far beyond the mountains, there lived a wise old king who had three sons. The eldest was strong and brave, the middle son was clever and cunning, and the youngest was kind and gentle. The king knew that his time was running short, and he needed to choose a successor who would rule the kingdom wisely. So he devised a test for his three sons. He called them to the great hall and said, I will send each of you on a quest. The one who returns with the greatest treasure shall inherit the throne. The eldest son rode north to the dragon mountains, seeking gold and jewels. The middle son sailed east across the great ocean, seeking ancient artifacts and magical weapons. The youngest son walked south into the humble villages, seeking nothing but wisdom and understanding from the common people. After a year and a day, the three sons returned to the castle. The eldest brought a chest overflowing with gold coins and precious gems taken from the dragon hoard. The middle son brought a ship laden with exotic treasures from distant lands and a sword said to be forged by the gods themselves. The youngest son brought nothing but a worn leather journal filled with stories, remedies, and farming techniques he had learned from the villagers." \
    --max_tokens 500

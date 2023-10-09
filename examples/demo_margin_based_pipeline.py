"""Demonstrate how to use the MarginBasedPipeline."""

import hadal

if __name__ == "__main__":
    model_name = "setu4993/LaBSE"

    source_test = [
        "I think I like wine now.",
        "She eats one apple every day.",
        "They serve pizza dogs in the cafeteria.",
        "Empty sentence.",
    ]

    target_test = [
        "Je pense que j'aime le vin maintenant.",
        "Elle mange une pomme chaque jour.",
        "Ils vendent des hot-dogs à la cafétéria.",
        "Ce jeu se joue sur le vaisseau spatial.",
        "Barry est plus tard déchargé du corps après sa dernière bataille.",
    ]

    alignment_config = hadal.MarginBasedPipeline(model_name_or_path=model_name, model_device="cpu", faiss_device="cpu")

    result = alignment_config.make_alignments(
        source_sentences=source_test,
        target_sentences=target_test,
        knn_neighbors=2,
    )

    for ind, (score, src, tgt) in enumerate(result, start=1):
        print(f"Pair: {ind}")
        print(f"Score: {score}\nSource: {src}\nTarget: {tgt}\n")

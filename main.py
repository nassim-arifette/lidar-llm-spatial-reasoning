import argparse
from pathlib import Path

from utils.logger import Logger


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Plateforme d'entraînement de Lidar LLM")
    parser.add_argument("--mode", type=str,
                        choices=["train", "distillation", "predict", "spatial"],
                        default="predict", help="Mode d'exécution")
    parser.add_argument("--data_path", type=str, default="random",
                        help="Chemin vers les données brutes ('random' pour générer un dataset aléatoire)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Répertoire pour sauvegarder les résultats")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers un point de contrôle pour reprendre l'entraînement ou l'inférence")
    parser.add_argument("--debug", action="store_true",
                        help="Activer le mode débogage")
    # Arguments propres au mode "spatial" (raisonnement spatial à perception fixée).
    parser.add_argument("--spatial_args", nargs=argparse.REMAINDER, default=[],
                        help="Arguments transmis tels quels à spatial_reasoning.run_experiments")
    return parser.parse_args()


def run_spatial(args, logger) -> int:
    """Comparaison « réponse apprise » vs « programme géométrique explicite ».

    Ce mode n'utilise ni GPT-2, ni LLaMA, ni GPU : il consomme des détections
    3D déjà extraites et compare quatre systèmes au même budget de perception.
    Les imports lourds (transformers, peft, torch_geometric) ne sont volontairement
    pas effectués ici, pour que ce mode reste exécutable sur une machine modeste.
    """
    from spatial_reasoning.run_experiments import main as spatial_main

    extra = list(args.spatial_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if not any(a.startswith("--output-dir") for a in extra):
        extra += ["--output-dir", str(Path(args.output_dir) / "spatial_reasoning")]

    logger.info(f"Mode spatial : arguments {extra}")
    return spatial_main(extra)


def run_lidar_llm(args, logger) -> int:
    """Pipeline d'origine : encodeur LiDAR + VAT + GPT-2 (LoRA)."""
    # Imports différés : ce sont eux qui exigent torch, transformers, peft et
    # torch_geometric. Les garder locaux permet d'utiliser --mode spatial sans
    # installer la pile complète.
    import torch
    from data.dataset import create_data_loaders, create_train_test_val_datasets
    from models.model import Lidar_LLM
    from training.trainer import Trainer
    from inference.predictor import Predictor
    from utils.visualization import Visualizer
    from distillation.distillation import main as distil_main

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if args.mode != "distillation":
        logger.info("Initialisation du modèle")
        model = Lidar_LLM()
        if args.data_path == "random":
            logger.info("Utilisation d'un dataset généré aléatoirement")
        else:
            logger.info(f"Utilisation du dataset LiDAR depuis {args.data_path}")
        train_dataset, test_dataset, val_dataset = create_train_test_val_datasets(
            model.lidar_encoder, args.data_path
        )
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
    else:
        model = None

    if args.checkpoint and model is not None:
        logger.info(f"Chargement du point de contrôle depuis {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    visualizer = Visualizer()

    if args.mode == "train":
        logger.info("Démarrage de l'entraînement du modèle")
        trainer = Trainer(
            model,
            torch.nn.CrossEntropyLoss(),
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            device,
            config=None,
        )
        model = trainer.fit(train_loader, val_loader, epochs=2)

        caption_metrics = trainer.evaluate_captioning(val_loader)
        logger.info(f"Captioning Metrics: {caption_metrics}")

        logger.info("Génération des visualisations d'entraînement")
        visualizer.plot_training_history(trainer.history)

    elif args.mode == "distillation":
        logger.info("Démarrage de l'entraînement du modèle distillé")
        distil_main(str(output_dir))

    elif args.mode == "predict":
        logger.info("Génération de prédictions")
        predictor = Predictor(model, device)
        predictions = predictor.predict(next(iter(val_loader)))
        print(predictions)

    return 0


def main() -> int:
    args = parse_arguments()

    logger = Logger("outputs", "main")
    logger.info("Démarrage de la plateforme LidarLLM")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "spatial":
        status = run_spatial(args, logger)
    else:
        status = run_lidar_llm(args, logger)

    logger.info(f"Exécution terminée. Résultats sauvegardés dans {args.output_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

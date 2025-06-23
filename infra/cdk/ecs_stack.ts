import {
  Stack,
  StackProps,
  Duration,
  RemovalPolicy,
  CfnOutput,
} from "aws-cdk-lib";
import { Construct } from "constructs";
import {
  Vpc,
  SubnetType,
  SecurityGroup,
} from "aws-cdk-lib/aws-ec2";
import {
  Cluster,
  TaskDefinition,
  ContainerImage,
  LogDriver,
  FargateService,
  Protocol,
  Compatibility,           // ← add
} from "aws-cdk-lib/aws-ecs";
import {
  Queue,
} from "aws-cdk-lib/aws-sqs";
import {
  Table,
  AttributeType,
} from "aws-cdk-lib/aws-dynamodb";
import {
  ApplicationLoadBalancer,
  ApplicationProtocol,
  ListenerAction,
  ApplicationTargetGroup,
  TargetType,
} from "aws-cdk-lib/aws-elasticloadbalancingv2";
import {
  Repository,
} from "aws-cdk-lib/aws-ecr";
import { Bucket, BucketEncryption } from "aws-cdk-lib/aws-s3";

export class EcsStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // VPC -------------------------------------------------------------------
    const vpc = new Vpc(this, "Vpc", {
      maxAzs: 2,
      subnetConfiguration: [
        { name: "public", subnetType: SubnetType.PUBLIC },
      ],
    });

    // ECS Cluster -----------------------------------------------------------
    const cluster = new Cluster(this, "Cluster", { vpc });

    // S3 artefact bucket ----------------------------------------------------
    const artifactsBucket = new Bucket(this, "ArtifactsBucket", {
      bucketName: "hc-artifacts",
      encryption: BucketEncryption.S3_MANAGED,
      removalPolicy: RemovalPolicy.RETAIN,      // keep data if stack deleted
    });

    // SQS queue -------------------------------------------------------------
    const queue = new Queue(this, "ArtifactsQueue", {
      queueName: "hc-artifacts-queue",
      visibilityTimeout: Duration.seconds(300), // 5 min, matches celery
      retentionPeriod: Duration.days(4),
    });

    // DynamoDB table --------------------------------------------------------
    const table = new Table(this, "RunsTable", {
      tableName: "hc-runs",
      partitionKey: { name: "runId", type: AttributeType.STRING },
      removalPolicy: RemovalPolicy.RETAIN,
    });

    // Task definition (shared image) ---------------------------------------
    const repo = Repository.fromRepositoryName(
      this,
      "RunnerRepo",
      "hc-runner"
    );

    const taskDef = new TaskDefinition(this, "TaskDef", {
      memoryMiB: "1024",
      cpu: "512",
      compatibility: Compatibility.FARGATE,
    });

    const container = taskDef.addContainer("AppContainer", {
      image: ContainerImage.fromEcrRepository(repo, "latest"),
      essential: true,
      logging: LogDriver.awsLogs({ streamPrefix: "runner" }),
      command: ["python", "-m", "hc_services.runner.app"],
      portMappings: [
        { containerPort: 8000, protocol: Protocol.TCP },   // ← add
      ],
    });

    // ALB + service ---------------------------------------------------------
    const alb = new ApplicationLoadBalancer(this, "Alb", {
      vpc,
      internetFacing: true,
    });

    const listener = alb.addListener("Https", {
      port: 80,
      protocol: ApplicationProtocol.HTTP, // demo; add HTTPS + cert later
      defaultAction: ListenerAction.fixedResponse(200, {
        contentType: "text/plain",
        messageBody: "OK",
      }),
    });

    const runnerSvc = new FargateService(this, "RunnerSvc", {
      cluster,
      taskDefinition: taskDef,
      desiredCount: 1,
    });

    listener.addTargets("RunnerTG", {
      port: 8000,                       // match containerPort
      protocol: ApplicationProtocol.HTTP,
      targets: [runnerSvc],
      healthCheck: { path: "/health" },
    });

    new CfnOutput(this, "AlbDNS", { value: alb.loadBalancerDnsName });
  }
}

import {
  Stack,
  StackProps,
  Duration,
  CfnOutput,
} from "aws-cdk-lib";
import { Construct } from "constructs";
import {
  Vpc,
  SubnetType,
} from "aws-cdk-lib/aws-ec2";
import {
  Cluster,
  TaskDefinition,
  ContainerImage,
  LogDriver,
  FargateService,
  Protocol,
  Compatibility,
} from "aws-cdk-lib/aws-ecs";
import { Queue } from "aws-cdk-lib/aws-sqs";
import { Table } from "aws-cdk-lib/aws-dynamodb";
import {
  ApplicationLoadBalancer,
  ApplicationProtocol,
  ListenerAction,
} from "aws-cdk-lib/aws-elasticloadbalancingv2";
import { Repository } from "aws-cdk-lib/aws-ecr";
import { Bucket } from "aws-cdk-lib/aws-s3";

export class EcsStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // ───────────────────────────────────────── VPC
    const vpc = new Vpc(this, "Vpc", {
      maxAzs: 2,
      subnetConfiguration: [{ name: "public", subnetType: SubnetType.PUBLIC }],
    });

    // ───────────────────────────────────────── ECS cluster
    const cluster = new Cluster(this, "Cluster", {
      vpc,
      clusterName: "artefact-context-cluster",
    });

    // ───────────────────────────────────────── Adopt existing resources
    const artifactsBucket = Bucket.fromBucketName(this, "ArtifactsBucket", "hc-artifacts");
    const table           = Table.fromTableName(this,  "RunsTable",      "hc-runs");

    // ───────────────────────────────────────── SQS queue
    const queue = new Queue(this, "ArtifactsQueue", {
      queueName: "hc-artifacts-queue",
      visibilityTimeout: Duration.seconds(300),
      retentionPeriod:   Duration.days(4),
    });

    // ───────────────────────────────────────── Shared task image
    const repo = Repository.fromRepositoryName(this, "RunnerRepo", "viewer-backend");

    // ============ Runner task-def (Flask) =================================
    const runnerTaskDef = new TaskDefinition(this, "RunnerTaskDef", {
      memoryMiB: "1024",
      cpu: "512",
      compatibility: Compatibility.FARGATE,
    });

    runnerTaskDef.addContainer("AppContainer", {
      image: ContainerImage.fromEcrRepository(repo, "latest"),
      essential: true,
      logging: LogDriver.awsLogs({ streamPrefix: "runner" }),
      command: ["python", "-m", "hc_services.runner.app"],
      portMappings: [{ containerPort: 8000, protocol: Protocol.TCP }],
    });

    // IAM – runner needs S3, DDB, **send** to SQS
    artifactsBucket.grantReadWrite(runnerTaskDef.taskRole);
    table.grantReadWriteData(runnerTaskDef.taskRole);
    queue.grantSendMessages(runnerTaskDef.taskRole);

    // ============ Worker task-def (Celery) ================================
    const workerTaskDef = new TaskDefinition(this, "WorkerTaskDef", {
      memoryMiB: "1024",
      cpu: "512",
      compatibility: Compatibility.FARGATE,
    });

    workerTaskDef.addContainer("AppContainer", {
      image: ContainerImage.fromEcrRepository(repo, "latest"),
      essential: true,
      logging: LogDriver.awsLogs({ streamPrefix: "worker" }),
      command: ["celery", "-A", "hc_services.runner.tasks", "worker", "--loglevel=info"],
    });

    // IAM – worker needs S3, DDB, **consume** SQS
    artifactsBucket.grantReadWrite(workerTaskDef.taskRole);
    table.grantReadWriteData(workerTaskDef.taskRole);
    queue.grantConsumeMessages(workerTaskDef.taskRole);

    // ───────────────────────────────────────── ALB
    const alb = new ApplicationLoadBalancer(this, "Alb", {
      vpc,
      internetFacing: true,
    });

    const listener = alb.addListener("Http", {
      port: 80,
      protocol: ApplicationProtocol.HTTP,
      defaultAction: ListenerAction.fixedResponse(200, {
        contentType: "text/plain",
        messageBody: "OK",
      }),
    });

    // ───────────────────────────────────────── Runner service
    const runnerSvc = new FargateService(this, "RunnerSvc", {
      cluster,
      taskDefinition: runnerTaskDef,
      desiredCount: 1,
      serviceName: "runner-svc",
      assignPublicIp: true,
    });

    listener.addTargets("RunnerTG", {
      port: 8000,
      protocol: ApplicationProtocol.HTTP,
      targets: [runnerSvc],
      healthCheck: { path: "/health" },
    });

    // ───────────────────────────────────────── Worker service
    const workerSvc = new FargateService(this, "WorkerSvc", {
      cluster,
      taskDefinition: workerTaskDef,
      desiredCount: 1,
      serviceName: "worker-svc",
      assignPublicIp: true,
    });

    // ───────────────────────────────────────── Outputs
    new CfnOutput(this, "AlbDNS",      { value: alb.loadBalancerDnsName });
    new CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new CfnOutput(this, "RunnerName",  { value: runnerSvc.serviceName });
    new CfnOutput(this, "WorkerName",  { value: workerSvc.serviceName });
  }
}

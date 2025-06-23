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

    // ------------------------------------------------------------------ #
    // Networking                                                          #
    // ------------------------------------------------------------------ #
    const vpc = new Vpc(this, "Vpc", {
      maxAzs: 2,
      subnetConfiguration: [
        { name: "public", subnetType: SubnetType.PUBLIC },
      ],
    });

    // ------------------------------------------------------------------ #
    // ECS cluster (stable, friendly name)                                 #
    // ------------------------------------------------------------------ #
    const cluster = new Cluster(this, "Cluster", {
      vpc,
      clusterName: "artefact-context-cluster",   // stays constant
    });

    // ------------------------------------------------------------------ #
    // Existing resources (adopt, don’t create)                            #
    // ------------------------------------------------------------------ #
    const artifactsBucket = Bucket.fromBucketName(
      this,
      "ArtifactsBucket",
      "hc-artifacts",
    );

    const table = Table.fromTableName(
      this,
      "RunsTable",
      "hc-runs",
    );

    // ------------------------------------------------------------------ #
    // SQS queue (safe to recreate)                                        #
    // ------------------------------------------------------------------ #
    const queue = new Queue(this, "ArtifactsQueue", {
      queueName: "hc-artifacts-queue",
      visibilityTimeout: Duration.seconds(300),
      retentionPeriod: Duration.days(4),
    });

    // ------------------------------------------------------------------ #
    // Task definition                                                     #
    // ------------------------------------------------------------------ #
    const repo = Repository.fromRepositoryName(
      this,
      "RunnerRepo",
      "viewer-backend",
    );

    const taskDef = new TaskDefinition(this, "TaskDef", {
      memoryMiB: "1024",
      cpu: "512",
      compatibility: Compatibility.FARGATE,
    });

    taskDef.addContainer("AppContainer", {
      image: ContainerImage.fromEcrRepository(repo, "latest"),
      essential: true,
      logging: LogDriver.awsLogs({ streamPrefix: "runner" }),
      command: ["python", "-m", "hc_services.runner.app"],
      portMappings: [{ containerPort: 8000, protocol: Protocol.TCP }],
    });

    // ------------------------------------------------------------------ #
    // ALB + Fargate service                                               #
    // ------------------------------------------------------------------ #
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

    const runnerSvc = new FargateService(this, "RunnerSvc", {
      cluster,
      taskDefinition: taskDef,
      desiredCount: 1,
      serviceName: "runner-svc",      // ← fixed, readable name
      assignPublicIp: true,           // quick way to reach ECR/S3
    });

    listener.addTargets("RunnerTG", {
      port: 8000,
      protocol: ApplicationProtocol.HTTP,
      targets: [runnerSvc],
      healthCheck: { path: "/health" },
    });

    // ------------------------------------------------------------------ #
    // Stack outputs                                                      #
    // ------------------------------------------------------------------ #
    new CfnOutput(this, "AlbDNS",   { value: alb.loadBalancerDnsName });
    new CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new CfnOutput(this, "ServiceName", { value: runnerSvc.serviceName });
  }
}

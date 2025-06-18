import { Stack, StackProps, Duration, RemovalPolicy } from "aws-cdk-lib";
import {
  Bucket,
  BucketAccessControl,
  BlockPublicAccess,
} from "aws-cdk-lib/aws-s3";
import {
  CloudFrontWebDistribution,
  OriginAccessIdentity,
} from "aws-cdk-lib/aws-cloudfront";
import { Construct } from "constructs";

export class S3SpaStack extends Stack {
  public readonly distributionDomain: string;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const spaBucket = new Bucket(this, "SpaBucket", {
      accessControl: BucketAccessControl.PRIVATE,
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const oai = new OriginAccessIdentity(this, "OAI");
    spaBucket.grantRead(oai);

    const distribution = new CloudFrontWebDistribution(
      this,
      "SpaDistribution",
      {
        originConfigs: [
          {
            s3OriginSource: { s3BucketSource: spaBucket, originAccessIdentity: oai },
            behaviors: [{ isDefaultBehavior: true }],
          },
        ],
        defaultRootObject: "index.html",
        errorConfigurations: [
          { errorCode: 403, responseCode: 200, responsePagePath: "/index.html" },
          { errorCode: 404, responseCode: 200, responsePagePath: "/index.html" },
        ],
      }
    );

    this.distributionDomain = distribution.distributionDomainName;
  }
}

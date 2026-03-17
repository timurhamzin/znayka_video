import tempfile
import unittest
from pathlib import Path

from subtitles_to_markdown import merge_subtitles


class MergeEncodingTest(unittest.TestCase):
    def test_merged_markdown_uses_specified_encoding_and_contains_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'Bebebears - TV cartoons'
            episode_18 = (
                root
                / 'Bebebears - Episode 18 - A trip to the stars - Kids Show - Super ToonsTV'
            )
            episode_19 = (
                root
                / 'Bebebears - Episode 19 - Dragon Heroes - Kids Show - Super ToonsTV'
            )
            episode_18.mkdir(parents=True, exist_ok=True)
            episode_19.mkdir(parents=True, exist_ok=True)
            (episode_18 / 'original').mkdir(parents=True, exist_ok=True)
            (episode_18 / 'translated_utf8').mkdir(parents=True, exist_ok=True)
            (episode_18 / 'translated_windows1251').mkdir(parents=True, exist_ok=True)

            srt_18 = episode_18 / (
                'Bebebears - A trip to the stars - Episode 18 - Kids Show - Super ToonsTV.srt'
            )
            srt_19 = episode_19 / (
                'Bebebears - Dragon Heroes - Episode 19 - Kids Show - Super ToonsTV.srt'
            )

            srt_18.write_text(
                '1\n00:00:01,000 --> 00:00:02,000\nПривет, серия 18.\n',
                encoding='windows-1251',
            )
            srt_19.write_text(
                '1\n00:00:03,000 --> 00:00:04,000\nПривет, серия 19.\n',
                encoding='windows-1251',
            )
            (episode_18 / 'original' / 'ep18.srt').write_text(
                '1\n00:00:05,000 --> 00:00:06,000\nORIGINAL EP18\n',
                encoding='windows-1251',
            )
            (episode_18 / 'translated_utf8' / 'ep18.srt').write_text(
                '1\n00:00:07,000 --> 00:00:08,000\nTRANSLATED UTF8 EP18\n',
                encoding='windows-1251',
            )
            (episode_18 / 'translated_windows1251' / 'ep18.srt').write_text(
                '1\n00:00:09,000 --> 00:00:10,000\nTRANSLATED WIN1251 EP18\n',
                encoding='windows-1251',
            )

            output_md = root / 'merged_srt_files.md'

            with self.assertLogs('subtitles_to_markdown', level='INFO') as logs:
                report = merge_subtitles(
                    source_dir=root,
                    output_md=output_md,
                    pattern='*.srt',
                    source_encoding='windows-1251',
                    output_encoding='windows-1251',
                )

            self.assertEqual(report.errors, [])
            self.assertGreaterEqual(len(report.warnings), 1)

            output_bytes = output_md.read_bytes()
            output_text = output_bytes.decode('windows-1251')

            with self.assertRaises(UnicodeDecodeError):
                output_bytes.decode('utf-8')

            self.assertIn('Episode 18', output_text)
            self.assertIn('Episode 19', output_text)
            self.assertIn(
                '## Bebebears - Episode 18 - A trip to the stars - Kids Show - Super ToonsTV',
                output_text,
            )
            self.assertIn(
                '## Bebebears - Episode 19 - Dragon Heroes - Kids Show - Super ToonsTV',
                output_text,
            )
            self.assertEqual(output_text.count('## Bebebears - Episode 18 '), 1)
            self.assertNotIn('ORIGINAL EP18', output_text)
            self.assertNotIn('TRANSLATED UTF8 EP18', output_text)
            self.assertNotIn('TRANSLATED WIN1251 EP18', output_text)
            self.assertIn('Привет, серия 18.', output_text)

            merged_logs = '\n'.join(logs.output)
            self.assertIn('Merge finished.', merged_logs)
            self.assertIn('Warnings (', merged_logs)
            self.assertIn('Errors (0):', merged_logs)

    def test_mixed_source_encodings_keep_episode_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'Bebebears - TV cartoons'
            episode_02 = root / 'Bebebears - Episode 02'
            episode_03 = root / 'Bebebears - Episode 03'
            episode_02.mkdir(parents=True, exist_ok=True)
            episode_03.mkdir(parents=True, exist_ok=True)

            (episode_02 / 'ep02.srt').write_text(
                '1\n00:00:01,000 --> 00:00:02,000\nEpisode 02 text\n',
                encoding='utf-8',
            )
            (episode_03 / 'ep03.srt').write_text(
                '1\n00:00:03,000 --> 00:00:04,000\nEpisode 03 text\n',
                encoding='windows-1251',
            )

            output_md = root / 'merged_srt_files.md'

            report = merge_subtitles(
                source_dir=root,
                output_md=output_md,
                pattern='*.srt',
                source_encoding='windows-1251',
                output_encoding='utf-8',
                source_fallback_encodings=['utf-8', 'utf-8-sig'],
            )

            self.assertEqual(report.errors, [])
            merged = output_md.read_text(encoding='utf-8')
            self.assertIn('Bebebears - Episode 02', merged)
            self.assertIn('Bebebears - Episode 03', merged)
            self.assertIn('Episode 02 text', merged)
            self.assertIn('Episode 03 text', merged)


if __name__ == '__main__':
    unittest.main()
